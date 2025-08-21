// langchain/agent.js
import {
  RunnableSequence,
  RunnableWithMessageHistory,
  RunnablePassthrough
} from '@langchain/core/runnables';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { performance } from 'perf_hooks';
import * as dotenv from 'dotenv';

import { getMemoryForUser } from './memory.js';
import { getRetrieverProvider } from '../database/retrieverFactory.js';

dotenv.config();

const llm = new ChatGoogleGenerativeAI({
  model: process.env.GOOGLE_LLM_MODEL || 'gemini-2.5-flash',
  temperature: 0.4,
  apiKey: process.env.GOOGLE_API_KEY
});

const chatPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    [
      "You are an advanced AI assistant that answers questions ONLY based on the retrieved documents.",
      "Rules:",
      "1) Use only the given context. If it's insufficient, say so clearly.",
      "2) No fabrication.",
      "3) Use the user's language.",
      "4) Keep it concise and structured.",
      "5) Use conversation history only for continuity, never to override facts.",
      "",
      "Context:",
      "{context}"
    ].join("\n")
  ],
  new MessagesPlaceholder("history"),
  ["human", "{question}"]
]);

export async function handleRAGChat({ userId, message }) {
  const startAll = performance.now();

  try {
    // Select retriever
   const retrieverProviders = process.env.RETRIEVER_PROVIDERS;
   console.log(`Using retriever providers: ${retrieverProviders}`);
   const retriever = await getRetrieverProvider(retrieverProviders.split(',')[0].trim());

    const t1 = performance.now();
    const memory = await getMemoryForUser(userId);
    console.log(`getMemoryForUser: ${(performance.now() - t1).toFixed(2)} ms`);

    // Create RAG chain
    const ragChain = RunnableSequence.from([
      RunnablePassthrough.assign({
        context: async (input) => {
          const tCtxStart = performance.now();
          const documents = await retriever.getRelevantDocuments(input.question);

          console.log(`[RAG] Retrieved ${documents.length} docs`);

          const ctx = documents.map((doc, i) => {
            const meta = doc.metadata || {};
            const tag = [
              meta.title ? `#${i + 1}: ${meta.title}` : `#${i + 1}`,
              meta.url ? `(${meta.url})` : "",
              typeof meta.score === "number" ? `score=${meta.score.toFixed(4)}` : "",
            ].filter(Boolean).join(" ");

            return `${tag}\n${doc.pageContent}`;
          }).join("\n\n---\n\n");

          console.log(`Retriever invoke: ${(performance.now() - tCtxStart).toFixed(2)} ms`);
          return ctx || "No relevant documents found.";
        },
        question: (input) => input.question,
      }),
      chatPrompt,
      llm,
      new StringOutputParser()
    ]);

    const chainWithMemory = new RunnableWithMessageHistory({
      runnable: ragChain,
      getMessageHistory: () => memory.chatHistory,
      inputMessagesKey: 'question',
      historyMessagesKey: 'history',
    });

    const t2 = performance.now();
    console.log(`LLM invoke with question: "${message}"`);
    const response = await chainWithMemory.invoke(
      { question: message },
      { configurable: { sessionId: userId } }
    );
    console.log(`LLM invoke: ${(performance.now() - t2).toFixed(2)} ms`);

    const t3 = performance.now();
    await memory.chatHistory.getMessages();
    console.log(`Retrieve updated messages: ${(performance.now() - t3).toFixed(2)} ms`);

    console.log(`Total time: ${(performance.now() - startAll).toFixed(2)} ms`);
    return response;

  } catch (error) {
    console.error('❌ Error in handleRAGChat:', error);
    return "I apologize, but I encountered an internal error while processing your request. Please try again shortly.";
  }
}
