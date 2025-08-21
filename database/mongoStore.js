// langchain/vectorStore.js

import { MongoClient } from 'mongodb';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Document } from '@langchain/core/documents';
import * as dotenv from 'dotenv';
dotenv.config();

let client;

export async function getMongoVectorStore() {
  const uri = process.env.MONGODB_URI;
  const dbName = process.env.MONGODB_NAME;
  const collectionName = 'gemini_documents';

  client ||= new MongoClient(uri);
  await client.connect();

  const db = client.db(dbName);
  const collection = db.collection(collectionName);

  const vectorStore = new MongoDBAtlasVectorSearch(
    new GoogleGenerativeAIEmbeddings({
      modelName: 'gemini-embedding-001',
      apiKey: process.env.GOOGLE_API_KEY,
    }),
    {
      collection,
      indexName: 'vector_index_gemini',
      textKey: 'text',
      embeddingKey: 'embedding',
    }
  );

  return vectorStore.asRetriever({
    k: 5,
    filter: {},
  });
}

