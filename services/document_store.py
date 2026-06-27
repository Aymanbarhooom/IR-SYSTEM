import json
import pymysql

from config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DATABASE,
    MYSQL_USER,
    MYSQL_PASSWORD,
    DOCUMENTS_FILE
)


class DocumentStore:

    def __init__(self):

        self.connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4"
        )

    def create_table(self):

        query = """
        CREATE TABLE IF NOT EXISTS documents (

            id VARCHAR(50) PRIMARY KEY,

            content LONGTEXT NOT NULL

        )
        """

        with self.connection.cursor() as cursor:

            cursor.execute(query)

        self.connection.commit()

    def import_documents(self):

        with open(
            DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

        total = len(documents)

        with self.connection.cursor() as cursor:

            for i, (doc_id, doc_data) in enumerate(
                documents.items(),
                start=1
            ):

                cursor.execute(
                    """
                    INSERT IGNORE INTO documents
                    (id, content)
                    VALUES (%s, %s)
                    """,
                    (
                        doc_id,
                        doc_data["text"]
                    )
                )

                if i % 5000 == 0:

                    print(
                        f"{i}/{total} documents imported"
                    )

        self.connection.commit()

        print(
            f"Imported {total} documents successfully"
        )

    def get_document(
        self,
        doc_id
    ):

        with self.connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT content
                FROM documents
                WHERE id = %s
                """,
                (doc_id,)
            )

            result = cursor.fetchone()

        return result