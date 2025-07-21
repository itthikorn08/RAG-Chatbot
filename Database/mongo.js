import { MongoClient} from 'mongodb';
import dotenv from 'dotenv';

dotenv.config();

const uri = process.env.MONGO_URI;
const client = new MongoClient(uri);

let db;

export async function connectMongo(dbName = 'rag_knowledge_base') {
    if (!db) {
        try {
            await client.connect();
            db = client.db(dbName);
            console.log("Connected to MongoDB");
        } catch (error) {
            console.error("Error connecting to MongoDB:", error);
            throw new Error("Could not connect to the database");
    }
    }
    return db;
    
}

export async function getCollection(collectionName) {
    if (!db) {
        await connectMongo();
    }
    return db.collection(collectionName);
}