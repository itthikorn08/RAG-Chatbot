// ESM
import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({
    url: 'http://10.99.2.14:6333/',
    apiKey: '52c8f9591d053f5bd169210c15e800f0bb449f0a1f24d985d432ac4665a9135b',
    // อย่าใส่ prefix ที่นี่!
    checkCompatibility: false,
});

const run = async () => {
    const res = await client.getCollections();
    console.log('collections:', res);
};
run().catch(console.error);