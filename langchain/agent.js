import { ChatOpenAI } from '@langchain/openai'; // ยังคงอยู่ หากใช้ LLM ตัวอื่นในส่วนอื่นๆ ของโปรเจกต์
import { RunnableSequence, RunnableWithMessageHistory } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { getMongoVectorStore } from './vectorStore.js';
import { getMemoryForUser } from './memory.js';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import * as dotenv from 'dotenv';
dotenv.config();

const LLM_CONTEXT_HISTORY_COUNT = 1;

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-2.5-flash', // แก้เป็น 1.5-flash ตามที่แนะนำก่อนหน้านี้ เพื่อความถูกต้องของชื่อโมเดล
    temperature: 0.7,
    apiKey: process.env.GOOGLE_API_KEY,
});

const prompt = PromptTemplate.fromTemplate(`
OBJECTIVE:
You are a helpful assistant that answers questions based only on documents the user is allowed to access.
Answer in the same language as the question.
Keep replies short and clear.
History: {history}
Context: {context}
Question: {question}
`);

export async function handleRAGChat({ userId, message }) {
    console.log('📩 Incoming message:', message);
    console.log('👤 From userId:', userId);

    try {
        const vectorStore = await getMongoVectorStore();
        const retriever = vectorStore.asRetriever({ k: 5 });

        const memory = await getMemoryForUser(userId);
        console.log('📚 Initializing memory for userId:', userId);
        console.log('📚 Chat history instance retrieved:', memory.chatHistory);

        const fullHistory = await memory.chatHistory.getMessages();
        //console.log(`📜 Full history retrieved from DB: ${fullHistory.length} messages.`);

        const slicedHistoryForLLM = fullHistory.slice(-LLM_CONTEXT_HISTORY_COUNT);
        //console.log(`📜 Sliced history for LLM context: ${slicedHistoryForLLM.length} messages.`);


        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const documents = await retriever.invoke(input.question);
                    // --- จุดที่ 1: ตรวจสอบเอกสารที่ดึงมา ---
                    console.log('📄 Documents retrieved by retriever:', documents.map(doc => ({
                        pageContent: doc.pageContent.substring(0, 100) + '...', // แสดงแค่ 100 ตัวอักษรแรก
                        metadata: doc.metadata
                    })));
                    // --- จบจุดที่ 1 ---

                    if (!Array.isArray(documents) || documents.length === 0) {
                        console.log('⚠️ No relevant documents found, context will be empty.'); // เพิ่ม log เมื่อไม่พบเอกสาร
                        return "No relevant documents found.";
                    }
                    const contextString = documents.map(doc => doc.pageContent).join('\n\n---\n\n');
                    // --- จุดที่ 2: ตรวจสอบ Context ที่จะส่งให้ LLM ---
                    console.log('📝 Prepared Context for LLM (first 500 chars):', contextString.substring(0, 500) + '...');
                    // --- จบจุดที่ 2 ---
                    return contextString;
                },
                question: (input) => input.question,
                history: (input) => slicedHistoryForLLM,
            },
            prompt,
            llm,
            new StringOutputParser(),
        ]);

        const chainWithMemory = new RunnableWithMessageHistory({
            runnable: ragChain,
            getMessageHistory: (sessionId) => memory.chatHistory,
            inputMessagesKey: 'question',
            historyMessagesKey: 'history',
        });

        const response = await chainWithMemory.invoke(
            { question: message },
            { configurable: { sessionId: userId } }
        );

        console.log('🗣️ Model Response:', response);

        const updatedMessages = await memory.chatHistory.getMessages();
        //console.log('📜 Updated chat history in memory:', updatedMessages.map(msg => ({ type: msg._getType(), content: msg.content })));

        return response;
    } catch (error) {
        console.error('❌ Error in handleRAGChat:', error);
        return "I apologize, but I encountered an internal error while processing your request. Please try again shortly.";
    }
}