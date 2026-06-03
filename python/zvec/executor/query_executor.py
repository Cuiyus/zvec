# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from _zvec import _Collection, _MultiQuery
from _zvec.param import _Fts, _SearchQuery, _SubQuery

from ..extension import ReRanker
from ..model.convert import convert_to_py_doc
from ..model.doc import QueryResult
from ..model.param.query import Query
from ..model.schema import CollectionSchema
from ..typing import DataType

__all__ = [
    "QueryContext",
    "QueryExecutor",
]

DTYPE_MAP = {
    DataType.VECTOR_FP16.value: np.float16,
    DataType.VECTOR_FP32.value: np.float32,
    DataType.VECTOR_FP64.value: np.float64,
    DataType.VECTOR_INT8.value: np.int8,
}


def convert_to_numpy(vec: Union[list, np.ndarray], dtype: np.dtype) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        if vec.dtype == dtype and vec.ndim == 1:
            return vec
        return np.asarray(vec, dtype=dtype).flatten()

    try:
        arr = np.asarray(vec, dtype=dtype)
        if arr.ndim != 1:
            arr = arr.flatten()
        return arr
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot convert input to 1D numpy array with dtype={dtype}: {type(vec)}"
        ) from e


class QueryContext:
    def __init__(
        self,
        topk: int,
        filter: Optional[str] = None,
        include_vector: bool = False,
        queries: Optional[list[Query]] = None,
        output_fields: Optional[list[str]] = None,
        reranker: Optional[ReRanker] = None,
    ):
        # query param
        self._filter = filter
        self._queries = queries or []
        self._topk = topk
        self._include_vector = include_vector
        self._output_fields = output_fields

        # reranker
        self._reranker = reranker

        # core vectors
        self._core_vectors = []

    @property
    def topk(self):
        return self._topk

    @property
    def queries(self):
        return self._queries

    @property
    def filter(self):
        return self._filter

    @property
    def reranker(self):
        return self._reranker

    @property
    def output_fields(self):
        return self._output_fields

    @property
    def include_vector(self):
        return self._include_vector

    @property
    def core_vectors(self):
        return self._core_vectors

    @core_vectors.setter
    def core_vectors(self, core_vectors: list[_SearchQuery]):
        self._core_vectors = core_vectors


class QueryExecutor:
    """Unified query executor that routes based on query count and reranker type."""

    def __init__(self, schema: CollectionSchema):
        self._schema = schema

    def _do_build(self, ctx: QueryContext, collection: _Collection) -> list[_SearchQuery]:
        """Build query vector list (no validation, conversion only)."""
        if not ctx.queries:
            return [self._do_build_query_wo_vector(ctx)]
        return [
            self._do_build_query_with_vector(ctx, query, collection)
            for query in ctx.queries
        ]

    def execute(self, ctx: QueryContext, collection: _Collection) -> QueryResult:
        """Execute query, selecting path based on conditions."""
        query_vectors = self._do_build(ctx, collection)
        if not query_vectors:
            raise ValueError("No query to execute")

        # Single query: direct SearchQuery path
        if len(query_vectors) == 1:
            docs = collection.Query(query_vectors[0])
            return [convert_to_py_doc(doc, self._schema) for doc in docs]

        # Multiple queries
        if ctx.reranker is not None and ctx.reranker._get_object() is None:
            # Python-only reranker: serial execution + Python rerank
            docs = self._execute_python_pipeline(query_vectors, collection)
            return self._do_merge_rerank_results(ctx, docs)

        # C++ MultiQuery path (with or without reranker)
        mvq = _MultiQuery()
        mvq.queries = [_SubQuery.from_search_query(vq) for vq in query_vectors]
        mvq.topk = ctx.topk
        if ctx.filter:
            mvq.filter = ctx.filter
        mvq.include_vector = ctx.include_vector
        if ctx.output_fields:
            mvq.output_fields = ctx.output_fields
        if ctx.reranker is not None:
            mvq.reranker = ctx.reranker._get_object()
        docs = collection.Query(mvq)
        return [convert_to_py_doc(doc, self._schema) for doc in docs]

    def _execute_python_pipeline(
        self, vectors: list[_SearchQuery], collection: _Collection
    ) -> list[QueryResult]:
        """Execute queries serially for Python-only reranker path."""
        results: list[QueryResult] = []
        for query in vectors:
            docs = collection.Query(query)
            results.append([convert_to_py_doc(doc, self._schema) for doc in docs])
        return results

    def _do_merge_rerank_results(
        self, ctx: QueryContext, docs_list: list[QueryResult]
    ) -> QueryResult:
        """Merge and rerank results from Python pipeline path."""
        if not docs_list:
            raise ValueError("Query results is empty")
        if len(docs_list) == 1 and not ctx.reranker:
            return docs_list[0]
        return ctx.reranker.rerank(docs_list)

    def _do_build_query_wo_vector(self, ctx: QueryContext) -> _SearchQuery:
        core_vector = _SearchQuery()
        core_vector.topk = ctx.topk
        core_vector.include_vector = ctx.include_vector
        if ctx.filter:
            core_vector.filter = ctx.filter
        if ctx.output_fields:
            core_vector.output_fields = ctx.output_fields
        return core_vector

    def _do_build_fts_query(self, query: Query, core_vector: _SearchQuery) -> None:
        """Set FTS query on core_vector if the query has FTS parameters."""
        if query.has_fts():
            fts = _Fts()
            fts.query_string = query.fts.query_string or ""
            fts.match_string = query.fts.match_string or ""
            core_vector.fts = fts

    def _do_build_query_with_vector(
        self, ctx: QueryContext, query: Query, collection: _Collection
    ) -> _SearchQuery:
        core_vector = self._do_build_query_wo_vector(ctx)
        core_vector.field_name = query.field_name
        if query.param:
            core_vector.query_params = query.param

        # set FTS query if provided
        self._do_build_fts_query(query, core_vector)

        # set output_fields
        core_vector.output_fields = ctx.output_fields

        vector_schema = None
        if query.has_vector() or query.has_id():
            vector_schema = (
                self._schema.vector(query.field_name)
                if query
                else self._schema.vectors[0]
            )

            if vector_schema is None:
                raise ValueError("No vector field found")

        # set vector
        if query.has_vector():
            vec_data = query.vector
        elif query.has_id():
            fetched = collection.Fetch([query.id])
            doc = next(iter(fetched.values()))
            if not doc:
                return core_vector
            vec_data = doc.get_any(vector_schema.name, vector_schema.data_type)
        else:
            return core_vector

        target_dtype = DTYPE_MAP.get(vector_schema.data_type.value)
        core_vector.set_vector(
            vector_schema._get_object(),
            convert_to_numpy(vec_data, target_dtype) if target_dtype else vec_data,
        )
        return core_vector
