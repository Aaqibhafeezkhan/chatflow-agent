import pytest

from chatflow_agent.channels.base import split_message


@pytest.mark.parametrize("max_len", [0, -1])
def test_split_message_rejects_non_positive_max_len(max_len):
    with pytest.raises(ValueError, match="max_len must be greater than 0"):
        split_message("Hello world", max_len=max_len)
