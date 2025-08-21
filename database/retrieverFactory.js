import { getMongoVectorStore } from './mongoStore.js';
import { getQdrantRetriever } from './qdrantStore.js';


const retrieverProviders = {
    qdrant: getQdrantRetriever,
    mongo: getMongoVectorStore,
};

export async function getRetrieverProvider(providerName) {
    const getRetriever = retrieverProviders[providerName?.toLowerCase()];
    if (!getRetriever) {
        throw new Error(`Unknown retriever provider: ${providerName}`);
    }
    return await getRetriever();
}
