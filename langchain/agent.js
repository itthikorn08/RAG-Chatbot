import { ChatOpenAI } from '@langchain/openai'; // ยังคงอยู่ หากใช้ LLM ตัวอื่นในส่วนอื่นๆ ของโปรเจกต์
import { RunnableSequence, RunnableWithMessageHistory } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { getMongoVectorStore } from './vectorStore.js';
import { getMemoryForUser } from './memory.js';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { performance } from 'perf_hooks'; // ✅ ใช้สำหรับจับเวลา
import * as dotenv from 'dotenv';
dotenv.config();

const LLM_CONTEXT_HISTORY_COUNT = 4;

// const llm = new ChatOpenAI({
//     model: 'gpt-4o-mini',
//     temperature: 0.4,
//     apiKey: process.env.OPENAI_API_KEY,
// });

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-2.5-flash',
    temperature: 0.4,
    apiKey: process.env.GOOGLE_API_KEY,
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
    console.log('Incoming message:', message);
    console.log('From userId:', userId);

    const startAll = performance.now();

    try {
        // ⏱ 1. โหลด Vector Store
        const t0 = performance.now();
        const vectorStore = await getMongoVectorStore();
        const retriever = vectorStore.asRetriever({ k: 5 });
        console.log(`getMongoVectorStore + retriever: ${(performance.now() - t0).toFixed(2)} ms`);

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
                    const documents = await retriever.invoke(input.question);
                    const contextString = (Array.isArray(documents) && documents.length > 0)
                        ? documents.map(doc => doc.pageContent).join('\n\n---\n\n')
                        : "No relevant documents found.";
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
