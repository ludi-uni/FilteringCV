import { useEffect, useRef, useState } from "react";
import type { ProgressRecord } from "../api/types";
import { jobWebSocketUrl } from "../api/client";

export type WsMessage =
  | ProgressRecord
  | { type: "ping" }
  | { type: "terminal"; status: string };

function isProgressRecord(msg: WsMessage): msg is ProgressRecord {
  return "job_id" in msg && "stage" in msg;
}

export function useJobWebSocket(jobId: string | null) {
  const [messages, setMessages] = useState<ProgressRecord[]>([]);
  const [terminalStatus, setTerminalStatus] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) {
      setMessages([]);
      setTerminalStatus(null);
      setConnected(false);
      return;
    }

    setMessages([]);
    setTerminalStatus(null);

    const socket = new WebSocket(jobWebSocketUrl(jobId));
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data as string) as WsMessage;
      if (isProgressRecord(data)) {
        setMessages((prev) => {
          const id = data.id ?? prev.length;
          const existing = prev.findIndex((m) => m.id === id);
          if (existing >= 0) {
            const next = [...prev];
            next[existing] = data;
            return next;
          }
          return [...prev, data];
        });
      } else if (data.type === "terminal") {
        setTerminalStatus(data.status);
      }
    };

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [jobId]);

  const latest = messages.length > 0 ? messages[messages.length - 1] : null;

  return { messages, latest, terminalStatus, connected };
}
