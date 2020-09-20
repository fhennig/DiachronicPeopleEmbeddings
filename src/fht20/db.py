import os
import psycopg2
import psycopg2.pool


class DBInfo:
    def __init__(self, host, port, username, password, db_name):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db_name = db_name

    def load_from_env():
        """Loads the database info that is stored in environment parameters."""
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        username = os.getenv("DB_USERNAME")
        password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        if username is None or port is None or password is None or db_name is None:
            raise Exception("Some information could not be loaded.")
        return DBInfo(host, port, username, password, db_name)

    def get_connection(self):
        """Returns a connection for this database."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.db_name
        )

    def get_connection_pool(self, max_connections):
        """Returns a threaded connection pool for this db."""
        return psycopg2.pool.ThreadedConnectionPool(
            1, max_connections,
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.db_name
        )        
