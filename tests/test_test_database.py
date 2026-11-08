from sqlalchemy import text


def test_suite_runs_against_the_app_test_database(db_engine):
    # The session fixture drops and recreates it, so no state survives from earlier sessions.
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one() == "app_test"
