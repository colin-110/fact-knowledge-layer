from app.pipeline.spatial_text import _group_into_rows, _group_rows_into_clusters


def _word(x0, y0, x1, y1, text):
    return (x0, y0, x1, y1, text, 0, 0, 0)


class TestGroupIntoRows:
    def test_words_on_same_line_form_one_row(self):
        words = [_word(50, 10, 70, 20, "Revenue"), _word(10, 10, 40, 20, "FY24")]
        rows = _group_into_rows(words, y_tolerance=3.0)
        assert len(rows) == 1
        # sorted left-to-right within the row despite input order
        assert [w[4] for w in rows[0]] == ["FY24", "Revenue"]

    def test_words_on_different_lines_form_separate_rows(self):
        words = [_word(10, 10, 40, 20, "Line1"), _word(10, 50, 40, 60, "Line2")]
        rows = _group_into_rows(words, y_tolerance=3.0)
        assert len(rows) == 2

    def test_small_y_jitter_within_tolerance_stays_one_row(self):
        words = [_word(10, 10.0, 40, 20, "A"), _word(50, 11.5, 80, 21, "B")]
        rows = _group_into_rows(words, y_tolerance=3.0)
        assert len(rows) == 1


class TestGroupRowsIntoClusters:
    def test_close_rows_form_one_cluster(self):
        rows = [[_word(10, 10, 40, 20, "A")], [_word(10, 25, 40, 35, "B")]]
        clusters = _group_rows_into_clusters(rows, cluster_gap=14.0)
        assert len(clusters) == 1

    def test_far_apart_rows_form_separate_clusters(self):
        # chart region ending far above a caption/footnote lower on the page
        rows = [[_word(10, 10, 40, 20, "ChartLabel")], [_word(10, 200, 40, 210, "Footnote")]]
        clusters = _group_rows_into_clusters(rows, cluster_gap=14.0)
        assert len(clusters) == 2

    def test_empty_rows_returns_empty(self):
        assert _group_rows_into_clusters([], cluster_gap=14.0) == []
