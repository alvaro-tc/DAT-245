import sqlite3

# Conectar a la base de datos (o crearla si no existe)
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Crear una tabla de ejemplo
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    rating REAL NOT NULL
)
''')

# Insertar datos de ejemplo
products = [
    ('Laptop', 'Electronics', 999.99, 4.5),
    ('Smartphone', 'Electronics', 699.99, 4.7),
    ('Tablet', 'Electronics', 299.99, 4.3),
    ('Headphones', 'Accessories', 199.99, 4.6),
    ('Smartwatch', 'Accessories', 199.99, 4.4)
]

cursor.executemany('''
INSERT INTO products (name, category, price, rating) VALUES (?, ?, ?, ?)
''', products)

conn.commit()


class SimpleReactiveAgent:
    def __init__(self, db_connection):
        self.conn = db_connection
    
    def query_database(self, category=None, max_price=None, min_rating=None):
        query = 'SELECT name, category, price, rating FROM products WHERE 1=1'
        params = []

        if category:
            query += ' AND category = ?'
            params.append(category)
        
        if max_price:
            query += ' AND price <= ?'
            params.append(max_price)
        
        if min_rating:
            query += ' AND rating >= ?'
            params.append(min_rating)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        return cursor.fetchall()

# Crear una instancia del agente
agent = SimpleReactiveAgent(conn)

# Realizar una búsqueda comparativa
results = agent.query_database(category='Electronics', max_price=800, min_rating=4.4)

# Imprimir los resultados
for result in results:
    print(f"Product: {result[0]}, Category: {result[1]}, Price: {result[2]}, Rating: {result[3]}")
