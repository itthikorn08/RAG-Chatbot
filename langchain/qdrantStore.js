// langchain/qdrantStore.js

import { QdrantClient } from "@qdrant/js-client-rest";
import { QdrantVectorStore } from "@langchain/qdrant";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from "dotenv";
dotenv.config();

const client = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GOOGLE_API_KEY,
});

export async function getQdrantVectorStore() {
    const collectionName = "ikb_content";
    const vectorName = "dense"; // ตั้งค่าชื่อเวกเตอร์ที่นี่

    // ตรวจสอบว่าคอลเลกชันมีอยู่หรือไม่
    const collections = await client.getCollections();
    const collectionExists = collections.collections.some(
        (c) => c.name === collectionName
    );
    
    if (!collectionExists) {
        throw new Error(`Collection '${collectionName}' does not exist.`);
    }

    // สร้าง VectorStore โดยระบุ queryVectorName
    const vectorStore = new QdrantVectorStore(embeddings, {
        client,
        collectionName,
        vectorName,
        queryVectorName: vectorName, // นี่คือส่วนสำคัญ
    });
    
    return vectorStore;
}