/**
 * Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of PiscesL1.
 * The PiscesL1 project belongs to the Dunimd Team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * DISCLAIMER: Users must comply with applicable AI regulations.
 * Non-compliance may result in service termination or legal liability.
 */

"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Mic, Video, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStore } from "@/lib/stores";
import { MainLayout } from "@/components/layout";
import type { ChatMessage, ContentPart } from "@/types";
import { cn } from "@/lib/utils";

interface MessageItemProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

function MessageItem({ message, isStreaming }: MessageItemProps) {
  const isUser = message.role === "user";
  
  const renderContent = () => {
    if (typeof message.content === "string") {
      return message.content;
    }
    
    return message.content.map((part, idx) => {
      switch (part.type) {
        case "text":
          return <span key={idx}>{part.text}</span>;
        case "image_url":
          return (
            <img
              key={idx}
              src={part.image_url?.url}
              alt="Image"
              className="max-w-md rounded-lg mt-2"
            />
          );
        case "audio_url":
          return (
            <div key={idx} className="flex items-center gap-2 rounded-lg bg-muted p-2 mt-2">
              <Mic className="h-4 w-4" />
              <span className="text-sm">Audio</span>
            </div>
          );
        case "video_url":
          return (
            <div key={idx} className="flex items-center gap-2 rounded-lg bg-muted p-2 mt-2">
              <Video className="h-4 w-4" />
              <span className="text-sm">Video</span>
            </div>
          );
        default:
          return null;
      }
    });
  };

  return (
    <div
      className={cn(
        "flex gap-3",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
        )}
      >
        {isUser ? "U" : "AI"}
      </div>
      <div
        className={cn(
          "flex-1 rounded-lg px-4 py-2",
          isUser ? "bg-primary/10" : "bg-muted"
        )}
      >
        <p className="whitespace-pre-wrap text-sm">
          {renderContent()}
          {isStreaming && (
            <Loader2 className="inline-block h-4 w-4 animate-spin ml-2" />
          )}
        </p>
      </div>
    </div>
  );
}

export default function InferencePage() {
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages, isLoading, sendMessageStream, currentModel } = useChatStore();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return;
    
    setInput("");
    setAttachments([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    await sendMessageStream(input);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setAttachments((prev) => [...prev, ...files]);
  };

  return (
    <MainLayout>
      <div className="flex h-full flex-col">
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full p-4">
            <div className="space-y-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center min-h-[400px]">
                  <div className="text-center">
                    <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                      <Send className="h-6 w-6 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-semibold">Start a conversation</h3>
                    <p className="text-sm text-muted-foreground">
                      Send a message to start chatting with PiscesL1
                    </p>
                  </div>
                </div>
              ) : (
                messages.map((message, idx) => (
                  <MessageItem
                    key={idx}
                    message={message}
                    isStreaming={
                      isLoading &&
                      idx === messages.length - 1 &&
                      message.role === "assistant"
                    }
                  />
                ))
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        </div>

        <div className="border-t p-4">
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <div className="flex items-center gap-2 rounded-lg border bg-background p-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Type a message..."
                  className="flex-1 bg-transparent border-0 focus:outline-none focus:ring-0 text-sm"
                />
              </div>
            </div>
            <Button
              onClick={handleSend}
              disabled={isLoading || (!input.trim() && attachments.length === 0)}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}
