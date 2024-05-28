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
	const DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
	const NO_SEED = -1
	inputFile := flag.String("input-file", "", "Path to the input text file")
	inputText := flag.String("input-text", "", "Input text to summarize")
	model := flag.String("model", "", "Model to use for the API")
	maxTokens := flag.Int("max-tokens", 200, "Maximum number of tokens in the summary")
	var seed int
	flag.IntVar(&seed, "seed", NO_SEED, "Seed for deterministic results (optional)")
	flag.Parse()

	var apiKey string
	apiKey = os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY not found in environment")
	}
	if *model == "" {
		*model = DEFAULT_OPENAI_MODEL
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()
	systemPrompt := `You are a text summarization assistant. 
	Generate a concise summary of the given input text while preserving the key information and main points. 
	Provide the summary in three bullet points, totalling 100 words or less.`

	var userMessage string
	if *inputFile != "" {
		content, err := os.ReadFile(*inputFile)
		if err != nil {
			log.Fatalf("Error reading input file: %v\n", err)
		}
		userMessage = string(content)
	} else if *inputText != "" {
		userMessage = *inputText
	} else {
		log.Fatal("Either input-file or input-text must be provided")
	}

	req := openai.ChatCompletionRequest{
		Model:       *model,
		MaxTokens:   *maxTokens,
		Stream:      true,
		Temperature: 0,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
			{Role: openai.ChatMessageRoleUser, Content: userMessage},
		},
	}

	if seed != NO_SEED {
		fmt.Printf("Using fixed seed: %d\n", seed)
		req.Seed = &seed
	}

	start := time.Now()
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

	elapsed := time.Since(start)
	fmt.Printf("\n\nTokens generated: %d\n", completionTokens)
	fmt.Printf("Output tokens per Second: %.2f/s\n", float64(completionTokens)/elapsed.Seconds())
	fmt.Printf("Total Execution Time: %s\n", elapsed)
}
