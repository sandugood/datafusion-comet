// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Per-executor-process cache for out-of-band Iceberg scan common data.
//!
//! For scans over large Iceberg MOR tables, the deduplication pools shared across a scan's
//! partitions (schemas, partition specs, delete-file lists, residuals, ...) can exceed protobuf's
//! hard 2 GiB single-message ceiling. Serializing them as one `IcebergScanCommon` message on the
//! JVM side threw `NegativeArraySizeException` from `AbstractMessageLite.toByteArray` (issue #4944).
//!
//! To avoid that, large scans shard the common data into multiple `IcebergScanCommon` chunks (each
//! well under the ceiling), register them once per executor via `Native.registerIcebergCommon`, and
//! reference the merged result from each per-task operator by `IcebergScan.common_ref`
//! (== the table `metadata_location`). This module owns the merge and the process-global cache.
//!
//! The merged value is a single `IcebergScanCommon` proto whose `repeated` pools are Rust `Vec`s
//! (bounded by `usize`, not by protobuf's message-size limit), reconstructed by concatenating each
//! chunk's pools in registration order. That order is exactly the flat pool-index space the
//! per-task `*_idx` fields reference, so downstream parsing is unchanged: the planner feeds the
//! merged common straight into `parse_file_scan_tasks_from_common`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use prost::Message;

use datafusion_comet_proto::spark_operator::IcebergScanCommon;

use crate::errors::CometError;

/// Soft cap on the number of distinct tables' common data retained per executor. A scan touches a
/// single table, so this bounds memory to the largest few tables scanned concurrently. Eviction is
/// insertion-order (not strict LRU) which is sufficient because keys are re-registered on demand by
/// the JVM guard in `CometExecRDD` whenever a lookup misses.
const MAX_CACHED_TABLES: usize = 32;

struct CommonCache {
    /// Insertion order of keys, for bounded eviction.
    order: Vec<String>,
    entries: HashMap<String, Arc<IcebergScanCommon>>,
}

impl CommonCache {
    fn new() -> Self {
        CommonCache {
            order: Vec::new(),
            entries: HashMap::new(),
        }
    }

    fn insert(&mut self, key: String, value: Arc<IcebergScanCommon>) {
        if self.entries.insert(key.clone(), value).is_none() {
            self.order.push(key);
        }
        while self.order.len() > MAX_CACHED_TABLES {
            let evicted = self.order.remove(0);
            self.entries.remove(&evicted);
        }
    }

    fn get(&self, key: &str) -> Option<Arc<IcebergScanCommon>> {
        self.entries.get(key).cloned()
    }

    fn remove(&mut self, key: &str) {
        if self.entries.remove(key).is_some() {
            self.order.retain(|k| k != key);
        }
    }
}

static CACHE: Lazy<Mutex<CommonCache>> = Lazy::new(|| Mutex::new(CommonCache::new()));

/// Merge `chunks` (each a serialized `IcebergScanCommon`) into one and register it under `key`.
///
/// Header/scalar fields (metadata_location, required_schema, catalog props, concurrency limit,
/// catalog_name) are taken from whichever chunk carries them; by construction the JVM sharder puts
/// them on the first chunk only. Every repeated pool is concatenated in chunk order.
pub fn register(key: &str, chunks: &[Vec<u8>]) -> Result<(), CometError> {
    if chunks.is_empty() {
        return Err(CometError::Internal(format!(
            "registerIcebergCommon called with no chunks for key '{key}'"
        )));
    }

    let mut merged = IcebergScanCommon::default();
    let mut header_taken = false;

    for (i, chunk) in chunks.iter().enumerate() {
        let part = IcebergScanCommon::decode(chunk.as_slice()).map_err(|e| {
            CometError::Internal(format!(
                "Failed to decode IcebergScanCommon chunk {i}/{} for key '{key}': {e}",
                chunks.len()
            ))
        })?;

        // Header fields: adopt from the first chunk that carries a non-empty metadata_location.
        if !header_taken && !part.metadata_location.is_empty() {
            merged.metadata_location = part.metadata_location.clone();
            merged.required_schema = part.required_schema.clone();
            merged.catalog_properties = part.catalog_properties.clone();
            merged.data_file_concurrency_limit = part.data_file_concurrency_limit;
            merged.catalog_name = part.catalog_name.clone();
            header_taken = true;
        }

        // Pools: concatenate in order to preserve the flat pool-index space.
        merged.schema_pool.extend(part.schema_pool);
        merged.partition_type_pool.extend(part.partition_type_pool);
        merged.partition_spec_pool.extend(part.partition_spec_pool);
        merged.name_mapping_pool.extend(part.name_mapping_pool);
        merged
            .project_field_ids_pool
            .extend(part.project_field_ids_pool);
        merged.partition_data_pool.extend(part.partition_data_pool);
        merged.delete_files_pool.extend(part.delete_files_pool);
        merged.residual_pool.extend(part.residual_pool);
    }

    if !header_taken {
        return Err(CometError::Internal(format!(
            "registerIcebergCommon chunks for key '{key}' carried no metadata_location header"
        )));
    }

    let mut cache = CACHE
        .lock()
        .map_err(|e| CometError::Internal(format!("Iceberg common cache lock poisoned: {e}")))?;
    cache.insert(key.to_string(), Arc::new(merged));
    Ok(())
}

/// Look up merged common by `key` (the `metadata_location` carried in `IcebergScan.common_ref`).
/// A miss means the JVM believed the key registered but the native cache evicted it; the caller
/// should surface a clear, retryable error so the JVM re-registers on the next attempt.
pub fn get(key: &str) -> Option<Arc<IcebergScanCommon>> {
    CACHE.lock().ok().and_then(|c| c.get(key))
}

/// Drop `key` from the cache (no-op if absent).
pub fn deregister(key: &str) {
    if let Ok(mut cache) = CACHE.lock() {
        cache.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_comet_proto::spark_operator::{DeleteFileList, IcebergDeleteFile};

    fn delete_list(path: &str) -> DeleteFileList {
        DeleteFileList {
            delete_files: vec![IcebergDeleteFile {
                file_path: path.to_string(),
                content_type: "POSITION_DELETES".to_string(),
                partition_spec_id: 0,
                equality_ids: vec![],
                file_size_in_bytes: 0,
            }],
        }
    }

    #[test]
    fn merges_pools_in_chunk_order() {
        let key = "test://table/merges_pools_in_chunk_order";

        // Chunk 0: header + first two delete-file-pool entries.
        let c0 = IcebergScanCommon {
            metadata_location: key.to_string(),
            data_file_concurrency_limit: 4,
            delete_files_pool: vec![delete_list("d0"), delete_list("d1")],
            ..Default::default()
        };
        // Chunk 1: header-less + two more pool entries.
        let c1 = IcebergScanCommon {
            delete_files_pool: vec![delete_list("d2"), delete_list("d3")],
            ..Default::default()
        };

        register(key, &[c0.encode_to_vec(), c1.encode_to_vec()]).unwrap();
        let merged = get(key).unwrap();

        assert_eq!(merged.metadata_location, key);
        assert_eq!(merged.data_file_concurrency_limit, 4);
        assert_eq!(merged.delete_files_pool.len(), 4);
        // Order preserved so pool index N still resolves to the Nth registered entry.
        let paths: Vec<&str> = merged
            .delete_files_pool
            .iter()
            .map(|l| l.delete_files[0].file_path.as_str())
            .collect();
        assert_eq!(paths, vec!["d0", "d1", "d2", "d3"]);

        deregister(key);
        assert!(get(key).is_none());
    }

    #[test]
    fn empty_chunks_is_error() {
        assert!(register("k", &[]).is_err());
    }
}
