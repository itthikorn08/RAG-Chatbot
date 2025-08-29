// database/qdrantStore.js
import { QdrantClient } from "@qdrant/js-client-rest";
import { Document } from "@langchain/core/documents";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from 'dotenv';

dotenv.config();

export class DenseQdrantRetriever {
    constructor({ client, collection, embeddings, k = 5, vectorName = "dense" }) {
        this.client = client;
        this.collection = collection;
        this.embeddings = embeddings;
        this.k = k;
        this.vectorName = vectorName;
    }

    async getRelevantDocuments(query) {
        const qvec = await this.embeddings.embedQuery(query);
        const res = await this.client.query(this.collection, {
            query: qvec,
            using: this.vectorName,
            limit: this.k,
            with_payload: true,
            with_vectors: false,
        });

        const pts = (res.points || res.scored_points || []);
        return pts
            .map(p => ({
                score: p.score,
                payload: p.payload || {},
            }))
            .filter(p => (p.payload?.clean_content || p.payload?.content || "").trim().length > 0)
            .map(p => new Document({
                pageContent: (p.payload.clean_content || p.payload.content || "").trim(),
                metadata: { ...p.payload, score: p.score },
            }));
    }
}

export async function getQdrantRetriever() {
    const client = new QdrantClient({
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
    });

    const embeddings = new GoogleGenerativeAIEmbeddings({
        modelName: process.env.GOOGLE_EMBED_MODEL || "gemini-embedding-001",
        apiKey: process.env.GOOGLE_API_KEY,
    });

    return new DenseQdrantRetriever({
        client,
        collection: process.env.QDRANT_COLLECTION || "ikb_content_gemini",
        embeddings,
        k: 5,
        vectorName: process.env.QDRANT_VECTOR_NAME || "dense",
    });
}
