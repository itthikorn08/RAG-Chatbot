import { ChatOpenAI } from '@langchain/openai';
import { RunnableSequence } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { getMongoVectorStore } from './vectorStore.js';

const llm = new ChatOpenAI({
    model: 'gpt-4o',
    temperature: 0.7,
});

const prompt = PromptTemplate.fromTemplate(`
ROLE / OBJECTIVE:
You are a helpful assistant that answers questions based only on documents the user is allowed to access.

Use only the accessible docs listed above.
If the user asks about restricted content, deny politely.
Do not use external knowledge, user IDs, or double quotes.

If the answer cannot be found in the documents, reply with:
"I'm sorry, I couldn't find that information in the knowledge base."

Answer in the same language as the question.
Keep replies short and clear.

Context:
{context}

Question: {question}
`);

export async function handleRAGChat({ userId, message }) {
    console.log('📩 Incoming message:', message);
    console.log('👤 From userId:', userId);

    const vectorStore = await getMongoVectorStore();

    const relevantDocs = await vectorStore.similaritySearch(message, 3);
    console.log('📚 Retrieved Documents:', relevantDocs);

    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');
    console.log('🧠 Combined Context:', context);

    const chain = RunnableSequence.from([
        {
            context: () => context,
            question: () => message,
        },
        prompt,
        llm,
    ]);

    const response = await chain.invoke({});
    console.log('🗣️ Model Response:', response);

    return response.content;
}
