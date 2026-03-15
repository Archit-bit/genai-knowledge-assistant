from app.evaluation import reciprocal_rank, source_recall_at_k, token_f1
from app.models import DocumentChunk, RetrievedChunk


def test_token_f1_returns_partial_overlap_score() -> None:
    score = token_f1(
        "Employees may work remotely up to three days.",
        "Employees may work remotely up to three days per week.",
    )
    assert 0.7 < score <= 1.0


def test_source_recall_and_rank_use_source_names() -> None:
    retrieved = [
        RetrievedChunk(
            chunk=DocumentChunk(
                id="remote_work_policy.md#chunk-0",
                source="remote_work_policy.md",
                text="Employees may work remotely up to three days per week.",
            ),
            score=0.91,
            rank=1,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(
                id="expense_policy.md#chunk-0",
                source="expense_policy.md",
                text="Expenses above $250 require pre-approval.",
            ),
            score=0.63,
            rank=2,
        ),
    ]

    assert source_recall_at_k(["remote_work_policy.md"], retrieved) == 1.0
    assert reciprocal_rank(["remote_work_policy.md"], retrieved) == 1.0
