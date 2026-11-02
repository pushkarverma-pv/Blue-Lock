from src.config import LEAGUES, PROJECT_ROOT, RANDOM_STATE, SEASONS


def test_project_configuration_is_importable():
    assert PROJECT_ROOT.exists()
    assert "E0" in LEAGUES
    assert len(SEASONS) >= 10
    assert RANDOM_STATE == 42
