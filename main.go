package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/sashabaranov/go-openai"
)

func main() {
	// This is fast, relatively small model from Google that summarises decently
	DEFAULT_OPEN_WEIGHTS_MODEL := "gemma:2b"
	IGNORED_OLLAMA_API_KEY := "ollama"

	// This is the ollama server installed from ollama.com
	DEFAULT_OLLAMA_SERVER_URL := "http://localhost:11434/v1"

	inputFile := flag.String("input-file", "", "Path to the input text file")
	inputText := flag.String("input-text", "", "Input text to summarize")
	model := flag.String("model", DEFAULT_OPEN_WEIGHTS_MODEL, "Model to use for the API")
	baseURL := flag.String("base-url", DEFAULT_OLLAMA_SERVER_URL, "Base URL for the Ollama server (which is OpenAI-compatible)")
	maxTokens := flag.Int("max-tokens", 200, "Maximum number of tokens in the summary")
	flag.Parse()

	// Define the system prompt
	systemPrompt := `You are a text summarization assistant. 
	Generate a concise summary of the given input text while preserving the key information and main points. 
	Provide the summary in three bullet points, totalling 100 words or less.`

	var userMessage string
	if *inputFile != "" {
		// Read input from file
		content, err := os.ReadFile(*inputFile)
		if err != nil {
			log.Fatalf("Error reading input file: %v\n", err)
		}
		userMessage = string(content)
	} else if *inputText != "" {
		// Use input text from command-line argument
		userMessage = *inputText
	} else {
		log.Fatal("Either input-file or input-text must be provided")
	}

	config := openai.DefaultConfig(IGNORED_OLLAMA_API_KEY)
	config.BaseURL = *baseURL

	client := openai.NewClientWithConfig(config)
	ctx := context.Background()
	start := time.Now()

	// We send a request to Ollama via the OpenAI protocol
	// in this example we don't do streaming, because the response will be generated very quickly particularly if you're using an M1+ Mac
	req := openai.ChatCompletionRequest{
		Model: *model,
		// these is the output token length
		MaxTokens: *maxTokens,
		Stream:    true,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userMessage,
			},
		},
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		log.Fatalf("ChatCompletionStream error: %v\n", err)
	}

	defer stream.Close()

	fmt.Printf("Summary: \n")

	var content string
	var completionTokens int
	for {
		response, err := stream.Recv()

		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalf("Stream error: %v\n", err)
		}

		content += response.Choices[0].Delta.Content
		fmt.Printf(response.Choices[0].Delta.Content)

		completionTokens += len(response.Choices[0].Delta.Content)
	}
	fmt.Printf("Summary: \n%s\n", content)

	elapsed := time.Since(start)
	fmt.Printf("\n\nTokens generated: %d\n", completionTokens)

	fmt.Printf("Output tokens per Second: %.2f\n", float64(completionTokens)/elapsed.Seconds())
	fmt.Printf("Total Execution Time: %s\n", elapsed)
}
