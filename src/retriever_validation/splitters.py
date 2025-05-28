from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                    SentenceTransformersTokenTextSplitter,
                                    TokenTextSplitter,
                                    # NLTKTextSplitter,
                                    SpacyTextSplitter
                                    )

TOKENS_PER_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# token_splitter = RecursiveCharacterTextSplitter(chunk_overlap=CHUNK_OVERLAP, chunk_size=TOKENS_PER_CHUNK_SIZE, separators=["\n\n", "\n", ".", " "],)
# splitter_1 = SentenceTransformersTokenTextSplitter(chunk_overlap=50,
#                                                        tokens_per_chunk=512,
#                                                        model_name='intfloat/multilingual-e5-large-instruct')
# splitter_1.name = 'sen_50_512'
# splitter_2 = SentenceTransformersTokenTextSplitter(chunk_overlap=0,
#                                                        tokens_per_chunk=512,
#                                                        model_name='intfloat/multilingual-e5-large-instruct')
# splitter_2.name = 'sen_0_512'

splitter_3 = RecursiveCharacterTextSplitter(separators=['\n', '\n\n', '.', '\n\n\n'],
                                                  chunk_overlap=50,
                                                  chunk_size=512)
splitter_3.name = 'rec_50_512'

splitter_4 = RecursiveCharacterTextSplitter(separators=['\n', '\n\n', '.', '\n\n\n'],
                                                  chunk_overlap=100,
                                                  chunk_size=1024)
splitter_4.name = 'rec_100_1024'

splitter_5 = RecursiveCharacterTextSplitter(separators=['\n', '\n\n', '.', '\n\n\n'],
                                                  chunk_overlap=0,
                                                  chunk_size=1024)
splitter_5.name = 'rec_0_1024'

splitter_6 = RecursiveCharacterTextSplitter(separators=['\n', '\n\n', '.', '\n\n\n'],
                                                  chunk_overlap=0,
                                                  chunk_size=2048)
splitter_6.name = 'rec_0_2048'

splitter_7 = TokenTextSplitter(             chunk_overlap=50,
                                                  chunk_size=512)
splitter_7.name = 'tok_50_512'

splitter_8 = TokenTextSplitter(             chunk_overlap=100,
                                                  chunk_size=1024)

splitter_8.name = 'tok_100_1024'

splitter_9 = TokenTextSplitter(             chunk_overlap=0,
                                              chunk_size=1024)

splitter_9.name = 'tok_0_1024'

splitter_10 = TokenTextSplitter(             chunk_overlap=0,
                                                  chunk_size=2048)

splitter_10.name = 'tok_0_2048'

splitter_11 = SpacyTextSplitter(
                                                  chunk_overlap=50,
                                                  chunk_size=512)

splitter_11.name = 'spac_50_512'

splitter_12 = SpacyTextSplitter(
                                                  chunk_overlap=100,
                                                  chunk_size=1024)
splitter_12.name = 'spac_100_1024'

splitter_13 = SpacyTextSplitter(
                                                  chunk_overlap=0,
                                                  chunk_size=1024)
splitter_13.name = 'spac_0_1024'

splitter_14 = SpacyTextSplitter(
                                                  chunk_overlap=0,
                                                  chunk_size=2048)
splitter_14.name = 'spac_0_2048'

# splitter_15 = NLTKTextSplitter(language='russian',
#                                                   chunk_overlap=50,
#                                                   chunk_size=512)

# splitter_15.name = 'nltk_50_512'

# splitter_16 = NLTKTextSplitter(language='russian',
#                                                   chunk_overlap=100,
#                                                   chunk_size=1024)
# splitter_16.name = 'nltk_100_1024'

# splitter_17 = NLTKTextSplitter(language='russian',
#                                                   chunk_overlap=0,
#                                                   chunk_size=1024)
# splitter_17.name = 'nltk_0_1024'

# splitter_18 = NLTKTextSplitter(language='russian',
#                                                   chunk_overlap=0,
# #                                                   chunk_size=2048)

# splitter_18.name = 'nltk_0_2048'


all_splitters = [splitter_3,splitter_4,splitter_5,
                 splitter_6,splitter_7,splitter_8,splitter_9,splitter_10,
                 splitter_11,splitter_12,splitter_13, splitter_14,]
