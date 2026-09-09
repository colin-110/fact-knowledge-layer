from app.pipeline.retrieval import _fts_query_string, rrf_fuse


class TestRRFFusion:
    def test_item_ranked_first_in_both_lists_wins(self):
        dense = ["a", "b", "c"]
        lexical = ["a", "c", "b"]
        fused = rrf_fuse([dense, lexical])
        assert fused[0][0] == "a"

    def test_item_only_in_one_list_still_included(self):
        dense = ["a", "b"]
        lexical = ["c"]
        fused = rrf_fuse([dense, lexical])
        ids = [doc_id for doc_id, _ in fused]
        assert set(ids) == {"a", "b", "c"}

    def test_item_appearing_in_both_lists_outranks_single_list_item(self):
        # "b" is rank 2 in both lists; "a" is rank 1 in dense only. RRF should still
        # let cross-list agreement compete with a single strong single-list rank.
        dense = ["a", "b"]
        lexical = ["z", "b"]
        fused = rrf_fuse([dense, lexical])
        scores = dict(fused)
        assert scores["b"] > scores["z"]

    def test_empty_lists(self):
        assert rrf_fuse([[], []]) == []


class TestFTSQueryString:
    def test_tokenizes_and_quotes(self):
        q = _fts_query_string("Delhivery FY24 revenue")
        assert '"Delhivery"' in q or '"delhivery"' in q.lower()
        assert " OR " in q

    def test_handles_percent_signs(self):
        q = _fts_query_string("60% growth")
        assert "60%" in q or "60" in q

    def test_empty_query_does_not_crash(self):
        q = _fts_query_string("")
        assert q == '""'
