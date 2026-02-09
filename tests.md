# curl tests

## Model list

```bash
curl http://localhost:8080/v1/models
```

## OpenAI non-streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Write a Python fibonacci function"}],"max_tokens":200,"temperature":1.0,"top_p":0.95}'
```

## OpenAI streaming

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Write a Python fibonacci function"}],"max_tokens":200,"stream":true,"temperature":1.0,"top_p":0.95}'
```

## Anthropic streaming

```bash
curl -N -X POST http://localhost:8080/v1/messages -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Write a Python fibonacci function"}],"max_tokens":200,"stream":true}'
```

## OpenAI streaming with tools

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"What is the weather in Tokyo?"}],"max_tokens":500,"stream":true,"temperature":1.0,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```
