// langchain/agent.js
import { RunnableSequence, RunnableWithMessageHistory } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { performance } from 'perf_hooks';
import * as dotenv from 'dotenv';
import { getMongoVectorStore } from './vectorStore.js';
import { getQdrantVectorStore } from './qdrantStore.js'; // นำเข้า Qdrant Vector Store
import { getMemoryForUser } from './memory.js';
import { QdrantClient } from "@qdrant/js-client-rest";
import { Document } from "@langchain/core/documents";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

dotenv.config();

// --- Configuration ---
const LLM_CONTEXT_HISTORY_COUNT = 4;
const USE_QDRANT = true; // <<< ปรับเป็น true เพื่อใช้ Qdrant หรือ false เพื่อใช้ MongoDB
const client = new QdrantClient({ url: 'http://ikb.exzycloud.com/qdrant', apiKey: '29bdfd0a1218d9729b7e93a6857f3755aee3a52018281657ec3871306da3e3fe' });

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-2.5-flash',
    temperature: 0.4,
    apiKey: process.env.GOOGLE_API_KEY,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "gemini-embedding-001",
    apiKey: process.env.GOOGLE_API_KEY,
    outputDimensionality: 768,
});

const prompt = PromptTemplate.fromTemplate(`
You are an advanced AI assistant that answers questions **only** based on the retrieved documents provided in the context.

Instructions:
1. Use only the given context to answer. If the context does not contain enough information, say so clearly.
2. Do not fabricate, guess, or add information from outside the context.
3. Use the same language as the user's question.
4. Keep answers clear, concise, and structured. Use bullet points or numbering if needed.
5. If there is historical conversation provided, use it to maintain continuity but do not override the factual accuracy from the context.

Context:
{context}

Conversation History:
{history}

User Question:
{question}

Answer:
`);

export async function handleRAGChat({ userId, message }) {
    // ...
    const startAll = performance.now();
    try {
        // ⏱ 1. โหลด Vector Store (เลือกใช้ตามการตั้งค่า)
        const t0 = performance.now();
        let vectorStore;

        if (USE_QDRANT) {
            console.log('Using Qdrant Vector Store...');
            vectorStore = await getQdrantVectorStore();
        } else {
            console.log('Using MongoDB Vector Store...');
            vectorStore = await getMongoVectorStore();
        }

        class DenseQdrantRetriever {
            constructor({ client, collection, embeddings, k = 5 }) {
                this.client = client;
                this.collection = collection;
                this.embeddings = embeddings;
                this.k = k;
            }
            async getRelevantDocuments(query) {
                const qvec = await this.embeddings.embedQuery(query);
                const res = await this.client.query(this.collection, {
                    query: qvec,
                    using: "dense", // 👈 บอกชื่อเวกเตอร์ให้ชัด
                    limit: this.k,
                    with_payload: true,
                });
                return res.points.map((p) =>
                    new Document({
                        pageContent: p.payload.clean_content ?? "",
                        metadata: p.payload,
                    })
                );
            }
        }

        // ใช้งาน
        const retriever = new DenseQdrantRetriever({
            client,
            collection: "ikb_content_gemini",
            embeddings,
            k: 5,
        });
        console.log(`Vector Store + retriever: ${(performance.now() - t0).toFixed(2)} ms`);


        // ⏱ 2. โหลด Memory
        const t1 = performance.now();
        const memory = await getMemoryForUser(userId);
        console.log(`getMemoryForUser: ${(performance.now() - t1).toFixed(2)} ms`);

        const fullHistory = await memory.chatHistory.getMessages();
        const slicedHistoryForLLM = fullHistory.slice(-LLM_CONTEXT_HISTORY_COUNT);

        // 3. สร้าง RAG Chain
        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const tCtxStart = performance.now();
                    const documents = await retriever.getRelevantDocuments(input.question);
                    console.log(`[DEBUG] Retrieved ${documents.length} documents for query: "${input.question}"`); // <--- เพิ่มบรรทัดนี้
                    // console.log('\n--- Retrieved Documents from Qdrant ---');
                    // if (documents && documents.length > 0) {
                    //     documents.forEach((doc, index) => {
                    //         console.log(`\nDocument ${index + 1}:`);
                    //         console.log('Content:', doc.pageContent.slice(0, 200) + '...'); // แสดงแค่บางส่วน
                    //         console.log('Metadata:', doc.metadata);
                    //     });
                    // } else {
                    //     console.log('No relevant documents found for this query.');
                    // }
                    // console.log('---------------------------------------\n');
                    const contextString = (Array.isArray(documents) && documents.length > 0)
                        ? documents.map(doc => doc.pageContent).join('\n\n---\n\n')
                        : "No relevant documents found.";

                    // console.log('\n--- Final Context to be sent to LLM ---');
                    // console.log(contextString);
                    // console.log('----------------------------------------\n');
                    console.log(`Retriever invoke: ${(performance.now() - tCtxStart).toFixed(2)} ms`);
                    return contextString;
                },
                question: (input) => input.question,
                history: (input) => slicedHistoryForLLM,
            },
            prompt,
            llm,
            new StringOutputParser(),
        ]);

        // 4. ทำงานพร้อม Memory
        const chainWithMemory = new RunnableWithMessageHistory({
            runnable: ragChain,
            getMessageHistory: (sessionId) => memory.chatHistory,
            inputMessagesKey: 'question',
            historyMessagesKey: 'history',
        });

        // ⏱ 5. เรียก LLM
        const t2 = performance.now();
        console.log(`LLM invoke with question: "${message}"`);
        const response = await chainWithMemory.invoke(
            { question: message },
            { configurable: { sessionId: userId } }
        );
        console.log(`LLM invoke: ${(performance.now() - t2).toFixed(2)} ms`);

        // ⏱ 6. อัปเดตประวัติแชท
        const t3 = performance.now();
        await memory.chatHistory.getMessages();
        console.log(`Retrieve updated messages: ${(performance.now() - t3).toFixed(2)} ms`);

        console.log(`Total time: ${(performance.now() - startAll).toFixed(2)} ms`);
        // console.log('Model Response:', response);

        return response;
    } catch (error) {
        console.error('❌ Error in handleRAGChat:', error);
        return "I apologize, but I encountered an internal error while processing your request. Please try again shortly.";
    }
}


