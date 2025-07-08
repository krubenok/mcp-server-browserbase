import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { 
  LLMClient, 
  AvailableModel, 
  ClientOptions, 
  LogLine 
} from "@browserbasehq/stagehand";
import { MCPSamplingLLMClient } from "./mcp-sampling-client.js";

/**
 * Custom LLM Provider that provides MCP Sampling clients when available,
 * falling back to creating LLM clients directly when not.
 */
export class MCPSamplingLLMProvider {
  private server: Server;
  private hasSamplingCapability: boolean;
  private logger: (message: LogLine) => void;

  constructor(
    server: Server,
    logger: (message: LogLine) => void,
    enableCaching: boolean = true,
    hasSamplingCapability: boolean = false
  ) {
    this.server = server;
    this.hasSamplingCapability = hasSamplingCapability;
    this.logger = logger;
  }

  /**
   * Get an LLM client, preferring MCP Sampling when available
   */
  getClient(modelName: AvailableModel, clientOptions?: ClientOptions): LLMClient {
    if (this.hasSamplingCapability) {
      // Use MCP Sampling client
      return new MCPSamplingLLMClient(this.server, modelName);
    } else {
      // This should not happen in our use case since we check capabilities before creating the provider
      // For now, return MCP sampling client as fallback
      return new MCPSamplingLLMClient(this.server, modelName);
    }
  }

  /**
   * Update sampling capability status
   */
  updateSamplingCapability(hasSamplingCapability: boolean) {
    this.hasSamplingCapability = hasSamplingCapability;
  }

  /**
   * Clean request cache (no-op since we don't have direct access to LLMProvider)
   */
  cleanRequestCache(requestId: string) {
    // No-op for now
  }
}