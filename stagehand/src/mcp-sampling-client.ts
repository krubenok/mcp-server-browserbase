import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CreateMessageRequestSchema,
  CreateMessageResultSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { 
  LLMClient, 
  CreateChatCompletionOptions, 
  LLMResponse, 
  ChatMessage,
  AvailableModel
} from "@browserbasehq/stagehand";

/**
 * LLM Client that uses MCP Sampling to request completions from the MCP client
 * instead of using a direct API key to an LLM provider.
 */
export class MCPSamplingLLMClient extends LLMClient {
  private server: Server;

  constructor(server: Server, modelName: AvailableModel = "gpt-4o", userProvidedInstructions?: string) {
    super(modelName, userProvidedInstructions);
    this.server = server;
    this.type = "mcp-sampling";
    this.hasVision = true; // Assume sampling client supports vision
  }

  async createChatCompletion<T = LLMResponse & { usage?: LLMResponse["usage"] }>(
    options: CreateChatCompletionOptions
  ): Promise<T> {
    const { messages, temperature, maxTokens, requestId } = options.options;
    const { logger } = options;

    try {
      logger({
        category: "llm",
        level: 1, // info
        message: `Requesting completion via MCP Sampling for request ${requestId}`,
        auxiliary: {
          messages: {
            value: messages.length.toString(),
            type: "integer"
          },
          temperature: {
            value: temperature?.toString() || "undefined",
            type: "float"
          },
          maxTokens: {
            value: maxTokens?.toString() || "undefined", 
            type: "integer"
          },
        },
      });

      // Convert Stagehand chat messages to MCP sampling format
      const mcpMessages = this.convertToMCPMessages(messages);

      // Create the sampling request
      const samplingRequest = {
        method: "sampling/createMessage" as const,
        params: {
          messages: mcpMessages,
          systemPrompt: this.getSystemPrompt(messages),
          modelPreferences: {
            hints: [{
              // This is a preference hint, not a requirement. The MCP client
              // ultimately decides which models are available and which to use.
              name: this.modelName,
            }],
            // Priority values optimized for web automation tasks:
            // - High intelligence (0.8) needed for understanding complex web pages and user interactions
            // - Moderate cost (0.5) as web automation often requires multiple LLM calls
            // - Moderate speed (0.5) as accuracy is more important than speed for reliable automation
            costPriority: 0.5,
            speedPriority: 0.5,
            intelligencePriority: 0.8,
          },
          maxTokens,
          temperature,
        },
      };

      logger({
        category: "llm",
        level: 2, // debug
        message: `Sending MCP sampling request`,
        auxiliary: {
          requestId: {
            value: requestId,
            type: "string"
          },
          messages: {
            value: mcpMessages.length.toString(),
            type: "integer"
          },
        },
      });

      // Send request to MCP client
      const result = await this.server.request(
        samplingRequest,
        CreateMessageResultSchema
      );

      logger({
        category: "llm",
        level: 1, // info
        message: `Received completion from MCP Sampling`,
        auxiliary: {
          requestId: {
            value: requestId,
            type: "string"
          },
          model: {
            value: (result as any).model || "unknown",
            type: "string"
          },
          stopReason: {
            value: (result as any).stopReason || "unknown",
            type: "string"
          },
        },
      });

      // Convert MCP result to LLMResponse format
      return this.convertFromMCPResult(result, requestId) as T;
    } catch (error) {
      logger({
        category: "llm",
        level: 0, // error
        message: `MCP Sampling request failed: ${error instanceof Error ? error.message : String(error)}`,
        auxiliary: {
          requestId: {
            value: requestId,
            type: "string"
          },
          error: {
            value: error instanceof Error ? error.message : String(error),
            type: "string"
          },
        },
      });
      throw error;
    }
  }

  /**
   * Convert Stagehand ChatMessage format to MCP SamplingMessage format
   */
  private convertToMCPMessages(messages: ChatMessage[]) {
    return messages
      .filter(msg => msg.role !== "system") // System messages handled separately
      .map(msg => ({
        role: msg.role as "user" | "assistant",
        content: this.convertMessageContent(msg.content),
      }));
  }

  /**
   * Convert message content to MCP format
   */
  private convertMessageContent(content: string | any[]): any {
    if (typeof content === "string") {
      return {
        type: "text",
        text: content,
      };
    }

    // Handle array of content (multimodal)
    if (Array.isArray(content)) {
      return content.map(item => {
        if (item.type === "text") {
          return {
            type: "text",
            text: item.text,
          };
        }
        if (item.type === "image_url" || item.image_url) {
          return {
            type: "image",
            data: item.image_url?.url || item.source?.data,
            mimeType: item.source?.media_type || "image/png",
          };
        }
        return item;
      });
    }

    return content;
  }

  /**
   * Extract system prompt from messages
   */
  private getSystemPrompt(messages: ChatMessage[]): string | undefined {
    const systemMessage = messages.find(msg => msg.role === "system");
    if (!systemMessage) return undefined;
    
    if (typeof systemMessage.content === "string") {
      return systemMessage.content;
    }
    
    // Extract text from multimodal content
    if (Array.isArray(systemMessage.content)) {
      const textContent = systemMessage.content
        .filter((item: any) => item.type === "text")
        .map((item: any) => item.text)
        .join("\n");
      return textContent || undefined;
    }
    
    return undefined;
  }

  /**
   * Convert MCP result to Stagehand LLMResponse format
   */
  private convertFromMCPResult(result: any, requestId: string): LLMResponse {
    const content = this.extractTextContent(result.content);
    
    return {
      id: requestId,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: result.model || this.modelName,
      choices: [{
        index: 0,
        message: {
          role: result.role || "assistant",
          content: content,
          tool_calls: [], // MCP sampling doesn't currently support tool calls in responses
        },
        finish_reason: this.mapStopReason(result.stopReason),
      }],
      usage: {
        prompt_tokens: 0, // MCP sampling doesn't provide token counts
        completion_tokens: 0,
        total_tokens: 0,
      },
    };
  }

  /**
   * Extract text content from MCP message content
   */
  private extractTextContent(content: any): string {
    if (typeof content === "string") {
      return content;
    }
    
    if (content?.type === "text") {
      return content.text;
    }
    
    if (Array.isArray(content)) {
      return content
        .filter((item: any) => item.type === "text")
        .map((item: any) => item.text)
        .join("");
    }
    
    return "";
  }

  /**
   * Map MCP stop reason to OpenAI format
   */
  private mapStopReason(stopReason?: string): string {
    switch (stopReason) {
      case "endTurn":
        return "stop";
      case "maxTokens":
        return "length";
      case "stopSequence":
        return "stop";
      default:
        return "stop";
    }
  }
}