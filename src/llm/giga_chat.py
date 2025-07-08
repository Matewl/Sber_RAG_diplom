from langchain_gigachat import GigaChat

class GigaChatLLM(GigaChat):
    def __init__(self, cfg):
        self.base_url = cfg['base_url']
        self.model = cfg['model_name']
        self.cert_file = cfg['cert_file']
        self.key_file = cfg['key_file']
        self.temperature = cfg['temperature']
        self.profanity_check = False
        self.top_p = cfg.get('top_p', 1)
        self.timeout = cfg.get('timeout', 30)
        self.verify_ssl_certs = False
