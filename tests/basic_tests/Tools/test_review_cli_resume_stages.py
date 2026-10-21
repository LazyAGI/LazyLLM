import pytest

from lazyllm.cli.review import _build_review_parser
from lazyllm.tools.git.review.checkpoint import ReviewStage


@pytest.mark.parametrize('stage', list(ReviewStage))
def test_current_review_stage_parses_and_converts(stage):
    args = _build_review_parser().parse_args(['--pr', '1', '--resume-from', stage.value])
    assert ReviewStage(args.resume_from) is stage


def test_resume_remains_optional():
    assert _build_review_parser().parse_args(['--pr', '1']).resume_from is None
