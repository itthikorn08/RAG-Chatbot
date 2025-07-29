import { ChatOpenAI } from '@langchain/openai';
import { RunnableSequence, RunnableWithMessageHistory } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { getMongoVectorStore } from './vectorStore.js';
import { getMemoryForUser } from './memory.js';

const LLM_CONTEXT_HISTORY_COUNT = 3; 

const llm = new ChatOpenAI({
    model: 'gpt-4o',
    temperature: 0.4,
});

const prompt = PromptTemplate.fromTemplate(`
ROLE / OBJECTIVE:
You are a helpful assistant that answers questions based only on documents the user is allowed to access.

Use only the accessible docs listed above.
If the user asks about restricted content, deny politely.


If the answer cannot be found in the documents, reply with:
"I'm sorry, I couldn't find that information in the knowledge base."

Answer in the same language as the question.
Keep replies short and clear.

History:
{history}

Context:
{context}

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
        console.log(`📜 Full history retrieved from DB: ${fullHistory.length} messages.`);
        
        const slicedHistoryForLLM = fullHistory.slice(-LLM_CONTEXT_HISTORY_COUNT);
        console.log(`📜 Sliced history for LLM context: ${slicedHistoryForLLM.length} messages.`);


        const ragChain = RunnableSequence.from([
            {
                context: async (input) => {
                    const documents = await retriever.invoke(input.question);
                    if (!Array.isArray(documents) || documents.length === 0) {
                        return "No relevant documents found."; 
                    }
                    return documents.map(doc => doc.pageContent).join('\n\n---\n\n');
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
        console.log('📜 Updated chat history in memory:', updatedMessages.map(msg => ({ type: msg._getType(), content: msg.content })));

        return response;
    } catch (error) {
        console.error('❌ Error in handleRAGChat:', error);
        return "I apologize, but I encountered an internal error while processing your request. Please try again shortly.";
    }
}