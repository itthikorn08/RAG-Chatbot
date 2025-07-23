// langchain/vectorStore.js

import { MongoClient } from 'mongodb';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { OpenAIEmbeddings } from '@langchain/openai';
import * as dotenv from 'dotenv';
dotenv.config();

let client;

export async function getMongoVectorStore() {
  const uri = process.env.MONGODB_URI;
  const dbName = process.env.MONGODB_NAME;
  const collectionName = 'documents';

  client ||= new MongoClient(uri);
  await client.connect();

  const db = client.db(dbName);
  const collection = db.collection(collectionName);

  // ✅ ใช้ constructor ตรงๆ แทน fromExistingIndex
  const vectorStore = new MongoDBAtlasVectorSearch(
    new OpenAIEmbeddings({
      model: 'text-embedding-3-small',
      openAIApiKey: process.env.OPENAI_API_KEY,
    }),
    {
      collection,
      indexName: 'vector_index',    // ชื่อ vector index ที่ตั้งไว้ใน Atlas
      textKey: 'text',       // ต้องตรงกับ field ใน collection
    }
  );

  return vectorStore;
}
