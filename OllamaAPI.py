import urllib3
import json


def request_ollama(data, api, url='127.0.0.1', port='11434'):
    """
    向Ollama发送请求
    :param data: 请求数据
    :param api: 请求的API
    :param url: 服务器地址
    :param port: 服务器端口
    :return: 服务器返回的数据
    """
    url = f'http://{url}:{port}/api/{api}'
    # 使用POST方法发送请求
    http = urllib3.PoolManager()
    response = http.request('POST',
                            url,
                            headers={'Content-Type': 'application/json'},
                            body=data)
    try:
        res = json.loads(response.data.decode('utf-8'))
    except Exception:
        res = response.data.decode('utf-8')
    return res


def get_ollama_embedding(text,
                         model='qwen2.5:14b',
                         url='127.0.0.1',
                         port='11434'):
    """
    获取ollama的embedding
    :param text: 输入文本
    :param model: 模型名称
    :param url: 服务器地址
    :param port: 服务器端口
    :return: embedding, 服务器返回的数据
    """
    data = json.dumps({"model": model, "input": text})
    res = request_ollama(data, 'embed', url, port)
    embedding = res['embeddings']
    if embedding.ndim == 2:
        embedding = embedding.squeeze()
    return embedding, res


def get_ollama_completion(prompt,
                          suffix=None,
                          system=None,
                          temperature=None,
                          top_p=None,
                          top_k=None,
                          seed=None,
                          context=None,
                          model='qwen2.5:14b',
                          url='127.0.0.1',
                          port=11434):
    """
    获取ollama的completion
    :param prompt: 输入文本
    :param suffix: 后缀文本 (可选)
    :param system: 系统信息 (可选)
    :param temperature: 温度 (可选)
    :param top_p: top_p (可选)
    :param top_k: top_k (可选)
    :param seed: 随机种子 (可选)
    :param context: 上下文 (可选)
    :param model: 模型名称
    :param url: 服务器地址
    :param port: 服务器端口
    :return: 服务器返回的数据
    """
    options = {}
    if temperature is not None:
        options['temperature'] = temperature
    if top_p is not None:
        options['top_p'] = top_p
    if top_k is not None:
        options['top_k'] = top_k
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options
    }
    if suffix is not None:
        data['suffix'] = suffix
    if seed is not None:
        data['options']['seed'] = seed
    if system is not None:
        data['system'] = system
    if context is not None:
        data['context'] = context
    data = json.dumps(data)
    res = request_ollama(data, 'generate', url, port)
    return res


class OllamaChatAgent:
    """
    聊天代理
    """

    def __init__(self,
                 system='',
                 model='qwen2.5:14b',
                 url='127.0.0.1',
                 port='11434'):
        """
        初始化
        :param system: 系统信息 (可选)
        :param model: 模型名称
        :param url: 服务器地址
        :param port: 服务器端口
        """
        self.model = model
        self.url = url
        self.port = port
        self.message = [{"role": "system", "content": system}]

    def chat(self,
             prompt,
             temperature=None,
             top_p=None,
             top_k=None,
             seed=None):
        """
        获取ollama的chat
        :param prompt: 输入文本
        :param temperature: 温度 (可选)
        :param top_p: top_p (可选)
        :param top_k: top_k (可选)
        :param seed: 随机种子 (可选)
        :return: 服务器返回的数据
        """
        self.message.append({"role": "user", "content": prompt})
        options = {}
        if temperature is not None:
            options['temperature'] = temperature
        if top_p is not None:
            options['top_p'] = top_p
        if top_k is not None:
            options['top_k'] = top_k
        data = {
            "model": self.model,
            "messages": self.message,
            "stream": False,
            "options": options
        }
        if seed is not None:
            data['options']['seed'] = seed
        data = json.dumps(data)
        res = request_ollama(data, 'chat', self.url, self.port)
        # 将聊天记录添加到message中
        self.message.append(res['message'])
        return res['message']['content']
