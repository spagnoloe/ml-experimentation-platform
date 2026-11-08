from sqlalchemy import text


def test_suite_runs_against_a_fresh_app_test_database(db_engine):
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one() == "app_test"
        tables = conn.execute(
            text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
        ).scalar_one()
    assert tables == 0
