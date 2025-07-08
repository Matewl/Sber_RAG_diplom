from langchain_gigachat import GigaChatEmbeddings

class Embedder(GigaChatEmbeddings):
    def __init__(self, cfg):
        self.base_url = cfg['base_url']
        self.model= cfg['model_name']
        self.cert_file=cfg['cert_file']
        self.key_file=cfg['key_file']
        self.verify_ssl_certs=False
