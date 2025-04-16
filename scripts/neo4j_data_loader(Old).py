from neo4j import GraphDatabase


# Neo4j connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

def execute_query(driver, query):
    with driver.session() as session:
        session.run(query)

def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Game) REQUIRE g.app_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ge:Genre) REQUIRE ge.name IS UNIQUE"
    ]

    for constraint in constraints:
        execute_query(driver, constraint)
        print(f"Applied constraint: {constraint}")

    load_queries = [

        # Load Game nodes
        """
        LOAD CSV WITH HEADERS FROM 'file:///games.csv' AS row
        CALL {
            WITH row
            MERGE (g:Game {app_id: row.app_id})
            ON CREATE SET 
              g.title = row.title,
              g.date_release = row.date_release,
              g.normalized_rating = toFloat(row.normalized_rating),
              g.win = (row.win = 'True'),
              g.mac = (row.mac = 'True'),
              g.linux = (row.linux = 'True')
        } IN TRANSACTIONS OF 1000 ROWS
        """,

        # Load Genre nodes
        """
        LOAD CSV WITH HEADERS FROM 'file:///genres.csv' AS row
        WITH row
        WHERE row.genre IS NOT NULL AND row.genre <> ""
        CALL {
            WITH row
            MERGE (ge:Genre {name: row.genre})
        } IN TRANSACTIONS OF 1000 ROWS
        """,

        # Load User nodes
        """
        LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
        WITH row
        WHERE row.user_id IS NOT NULL AND row.user_id <> ""
        CALL {
            WITH row
            MERGE (u:User {user_id: toInteger(row.user_id)})
        } IN TRANSACTIONS OF 1000 ROWS
        """,

        # Load HAS_GENRE relationships
        """
        LOAD CSV WITH HEADERS FROM 'file:///game_genre.csv' AS row
        WITH row
        WHERE row.genre IS NOT NULL AND row.genre <> "" AND row.app_id IS NOT NULL AND row.app_id <> ""
        CALL {
            WITH row
            MATCH (g:Game {app_id: row.app_id})
            MATCH (ge:Genre {name: row.genre})
            MERGE (g)-[:HAS_GENRE]->(ge)
        } IN TRANSACTIONS OF 1000 ROWS
        """,

        # Load LIKES relationships
        """
        LOAD CSV WITH HEADERS FROM 'file:///likes.csv' AS row
        WITH row
        WHERE row.user_id IS NOT NULL AND row.user_id <> "" AND row.app_id IS NOT NULL AND row.app_id <> ""
        CALL {
            WITH row
            MATCH (u:User {user_id: toInteger(row.user_id)})
            MATCH (g:Game {app_id: row.app_id})
            MERGE (u)-[r:LIKES]->(g)
            ON CREATE SET 
              r.playtime = toInteger(row.playtime),
              r.normalized_rating = toFloat(row.normalized_rating)
        } IN TRANSACTIONS OF 1000 ROWS
        """
    ]

    for idx, query in enumerate(load_queries, start=1):
        print(f"Running data load query {idx}...")
        execute_query(driver, query)
        print(f"Finished query {idx} ✅")

    driver.close()
    print("All data imported successfully 🎉")

if __name__ == "__main__":
    main()
