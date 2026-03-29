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

import { useState, useCallback, useEffect, MouseEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useApps } from "./apps-context";
import { useContextMenu } from "./context-menu";
import { AppWindow } from "./app-window";
import { useExplorerStore } from "@/lib/stores";
import {
  Folder,
  File,
  ChevronRight,
  HardDrive,
  ArrowUp,
  RefreshCw,
  Search,
  Grid,
  List,
  FolderOpen,
  FileText,
  Image,
  Video,
  Music,
  Archive,
  Code,
  FilePlus,
  FolderPlus,
  Computer,
  Copy,
  Scissors,
  Clipboard,
  Trash2,
  Edit,
  Info,
  Eye,
  Terminal,
  X,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface FileExplorerWindowProps {
  state: "minimized" | "normal" | "maximized";
}

interface FileItem {
  name: string;
  path: string;
  type: "file" | "directory" | "unknown";
  size?: number;
  modified?: string;
}

interface DriveInfo {
  name: string;
  path: string;
  total: number;
  used: number;
  free: number;
}

interface DirectoryInfo {
  path: string;
  items: FileItem[];
  disk?: {
    total: number;
    used: number;
    free: number;
  } | null;
  is_windows: boolean;
}

interface DrivesInfo {
  is_windows: boolean;
  drives: DriveInfo[];
}

const getFileIcon = (item: FileItem) => {
  if (item.type === "directory") {
    return <Folder className="h-5 w-5 text-amber-500" />;
  }
  
  const ext = item.name.split(".").pop()?.toLowerCase() || "";
  
  switch (ext) {
    case "txt":
    case "md":
    case "doc":
    case "docx":
      return <FileText className="h-5 w-5 text-blue-500" />;
    case "png":
    case "jpg":
    case "jpeg":
    case "gif":
    case "svg":
    case "webp":
      return <Image className="h-5 w-5 text-green-500" />;
    case "mp4":
    case "avi":
    case "mov":
    case "mkv":
      return <Video className="h-5 w-5 text-purple-500" />;
    case "mp3":
    case "wav":
    case "flac":
    case "aac":
      return <Music className="h-5 w-5 text-pink-500" />;
    case "zip":
    case "tar":
    case "gz":
    case "rar":
    case "7z":
      return <Archive className="h-5 w-5 text-orange-500" />;
    case "py":
    case "js":
    case "ts":
    case "jsx":
    case "tsx":
    case "java":
    case "c":
    case "cpp":
    case "rs":
    case "go":
      return <Code className="h-5 w-5 text-cyan-500" />;
    default:
      return <File className="h-5 w-5 text-gray-500" />;
  }
};

const formatSize = (bytes: number | undefined): string => {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
};

const formatDate = (dateStr: string | undefined): string => {
  if (!dateStr) return "-";
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return dateStr;
  }
};

export function FileExplorerWindow({ state }: FileExplorerWindowProps) {
  const { 
    minimizeApp, 
    closeApp, 
    maximizeApp, 
    restoreApp, 
    isAppMaximized, 
    getAppPosition, 
    updateAppPosition, 
    getAppSize, 
    updateAppSize,
    focusApp,
    isAppFocused,
  } = useApps();
  const { showContextMenu } = useContextMenu();
  const queryClient = useQueryClient();
  
  const savedPosition = getAppPosition("explorer");
  const savedSize = getAppSize("explorer");
  const maximized = isAppMaximized("explorer");
  const focused = isAppFocused("explorer");
  
  const [currentPath, setCurrentPath] = useState<string>("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [showDrivesView, setShowDrivesView] = useState(true);
  const [clipboard, setClipboard] = useState<{ path: string; operation: "copy" | "cut" } | null>(null);
  const [renamingItem, setRenamingItem] = useState<FileItem | null>(null);
  const [newName, setNewName] = useState("");
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [creatingFile, setCreatingFile] = useState(false);
  const [newItemName, setNewItemName] = useState("");

  const { diskInfo, isLoading: drivesLoading, isWsConnected, connectWebSocket, disconnectWebSocket, browse } = useExplorerStore();

  const isWindows = diskInfo?.is_windows ?? true;

  useEffect(() => {
    if (!isWsConnected) {
      connectWebSocket();
    }
    return () => {
      disconnectWebSocket();
    };
  }, [isWsConnected, connectWebSocket, disconnectWebSocket]);

  const { items, isLoading: dirLoading } = useExplorerStore();

  const navigateTo = useCallback((path: string) => {
    if (path === "") {
      setShowDrivesView(true);
      setCurrentPath("");
    } else {
      setShowDrivesView(false);
      setCurrentPath(path);
    }
    setSelectedPath(null);
    setRenamingItem(null);
    setCreatingFolder(false);
    setCreatingFile(false);
  }, []);

  const goUp = useCallback(() => {
    if (showDrivesView) return;
    
    if (!currentPath) {
      setShowDrivesView(true);
      return;
    }
    
    if (isWindows) {
      const normalizedPath = currentPath.replace(/\\/g, "/");
      const parts = normalizedPath.split("/").filter(Boolean);
      
      if (parts.length <= 1) {
        setShowDrivesView(true);
        setCurrentPath("");
        return;
      }
      
      parts.pop();
      if (parts.length === 1 && parts[0].endsWith(":")) {
        navigateTo(parts[0] + "/");
      } else {
        navigateTo(parts.join("/"));
      }
    } else {
      if (currentPath === "/") {
        setShowDrivesView(true);
        setCurrentPath("");
        return;
      }
      const parts = currentPath.split("/").filter(Boolean);
      parts.pop();
      const newPath = parts.length === 0 ? "/" : "/" + parts.join("/");
      navigateTo(newPath);
    }
  }, [currentPath, isWindows, navigateTo, showDrivesView]);

  const canGoUp = useCallback(() => {
    if (showDrivesView) return false;
    if (!currentPath) return false;
    return true;
  }, [currentPath, showDrivesView]);

  const pathParts = currentPath ? currentPath.replace(/\\/g, "/").split("/").filter(Boolean) : [];

  const filteredItems = items?.filter((item) =>
    item.name.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const isLoading = drivesLoading || dirLoading;

  const refreshDirectory = useCallback(() => {
    browse(currentPath);
  }, [browse, currentPath]);

  const handleDelete = useCallback(async (item: FileItem) => {
    if (!confirm(`Are you sure you want to delete "${item.name}"?`)) return;
    
    const result = await apiClient.deleteItem(item.path);
    if (result.success) {
      refreshDirectory();
    } else {
      alert("Failed to delete: " + result.error);
    }
  }, [refreshDirectory]);

  const handleRename = useCallback(async () => {
    if (!renamingItem || !newName.trim()) {
      setRenamingItem(null);
      return;
    }
    
    const parentPath = renamingItem.path.substring(0, renamingItem.path.lastIndexOf("/"));
    const newPath = parentPath + "/" + newName.trim();
    
    const result = await apiClient.renameItem(renamingItem.path, newPath);
    if (result.success) {
      setRenamingItem(null);
      setNewName("");
      refreshDirectory();
    } else {
      alert("Failed to rename: " + result.error);
    }
  }, [renamingItem, newName, refreshDirectory]);

  const handleCreateFolder = useCallback(async () => {
    if (!newItemName.trim() || !currentPath) return;
    
    const folderPath = currentPath + (currentPath.endsWith("/") ? "" : "/") + newItemName.trim();
    
    const result = await apiClient.createFolder(folderPath);
    if (result.success) {
      setCreatingFolder(false);
      setNewItemName("");
      refreshDirectory();
    } else {
      alert("Failed to create folder: " + result.error);
    }
  }, [newItemName, currentPath, refreshDirectory]);

  const handleCreateFile = useCallback(async () => {
    if (!newItemName.trim() || !currentPath) return;
    
    const filePath = currentPath + (currentPath.endsWith("/") ? "" : "/") + newItemName.trim();
    
    const result = await apiClient.createFile(filePath);
    if (result.success) {
      setCreatingFile(false);
      setNewItemName("");
      refreshDirectory();
    } else {
      alert("Failed to create file: " + result.error);
    }
  }, [newItemName, currentPath, refreshDirectory]);

  const handlePaste = useCallback(async () => {
    if (!clipboard || !currentPath) return;
    
    const itemName = clipboard.path.substring(clipboard.path.lastIndexOf("/") + 1);
    const destPath = currentPath + (currentPath.endsWith("/") ? "" : "/") + itemName;
    
    if (clipboard.operation === "copy") {
      const result = await apiClient.copyItem(clipboard.path, destPath);
      if (result.success) {
        refreshDirectory();
      } else {
        alert("Failed to copy: " + result.error);
      }
    } else {
      const result = await apiClient.moveItem(clipboard.path, destPath);
      if (result.success) {
        setClipboard(null);
        refreshDirectory();
      } else {
        alert("Failed to move: " + result.error);
      }
    }
  }, [clipboard, currentPath, refreshDirectory]);

  const startRename = useCallback((item: FileItem) => {
    setRenamingItem(item);
    setNewName(item.name);
  }, []);

  const handleContextMenu = useCallback((e: MouseEvent, item?: FileItem) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (item) {
      setSelectedPath(item.path);
    }
    
    const menuItems = item ? [
      {
        id: "open",
        label: item.type === "directory" ? "Open" : "Open file",
        icon: item.type === "directory" ? <Folder className="h-4 w-4" /> : <Eye className="h-4 w-4" />,
        onClick: () => {
          if (item.type === "directory") {
            navigateTo(item.path);
          }
        },
      },
      { id: "divider1", label: "", divider: true },
      {
        id: "copy",
        label: "Copy",
        icon: <Copy className="h-4 w-4" />,
        shortcut: "Ctrl+C",
        onClick: () => {
          setClipboard({ path: item.path, operation: "copy" });
        },
      },
      {
        id: "cut",
        label: "Cut",
        icon: <Scissors className="h-4 w-4" />,
        shortcut: "Ctrl+X",
        onClick: () => {
          setClipboard({ path: item.path, operation: "cut" });
        },
      },
      { id: "divider2", label: "", divider: true },
      {
        id: "rename",
        label: "Rename",
        icon: <Edit className="h-4 w-4" />,
        shortcut: "F2",
        onClick: () => {
          startRename(item);
        },
      },
      {
        id: "delete",
        label: "Delete",
        icon: <Trash2 className="h-4 w-4" />,
        danger: true,
        shortcut: "Del",
        onClick: () => {
          handleDelete(item);
        },
      },
      { id: "divider3", label: "", divider: true },
      {
        id: "properties",
        label: "Properties",
        icon: <Info className="h-4 w-4" />,
        onClick: () => {
          console.log("Properties:", item.path);
        },
      },
    ] : [
      {
        id: "refresh",
        label: "Refresh",
        icon: <RefreshCw className="h-4 w-4" />,
        onClick: () => refreshDirectory(),
      },
      { id: "divider1", label: "", divider: true },
      {
        id: "newFolder",
        label: "New Folder",
        icon: <FolderPlus className="h-4 w-4" />,
        onClick: () => {
          setCreatingFolder(true);
          setCreatingFile(false);
          setNewItemName("");
        },
      },
      {
        id: "newFile",
        label: "New File",
        icon: <FilePlus className="h-4 w-4" />,
        onClick: () => {
          setCreatingFile(true);
          setCreatingFolder(false);
          setNewItemName("");
        },
      },
      ...(clipboard && !showDrivesView ? [
        { id: "divider2", label: "", divider: true },
        {
          id: "paste",
          label: "Paste",
          icon: <Clipboard className="h-4 w-4" />,
          shortcut: "Ctrl+V",
          onClick: () => {
            handlePaste();
          },
        },
      ] : []),
      { id: "divider3", label: "", divider: true },
      {
        id: "openTerminal",
        label: "Open in Terminal",
        icon: <Terminal className="h-4 w-4" />,
        onClick: () => {
          console.log("Open terminal at:", currentPath);
        },
      },
    ];
    
    showContextMenu(e.clientX, e.clientY, menuItems);
  }, [showContextMenu, navigateTo, refreshDirectory, currentPath, clipboard, showDrivesView, handleDelete, startRename, handlePaste]);

  const handleBackgroundContextMenu = useCallback((e: MouseEvent) => {
    handleContextMenu(e);
  }, [handleContextMenu]);

  if (state === "minimized") {
    return null;
  }

  return (
    <AppWindow
      appId="explorer"
      title="File Explorer"
      icon={<Folder className="h-5 w-5 text-amber-500" />}
      defaultSize={{ width: 1000, height: 700 }}
      onMinimize={() => minimizeApp("explorer")}
      onClose={() => closeApp("explorer")}
      savedPosition={savedPosition}
      onPositionChange={(pos) => updateAppPosition("explorer", pos)}
      savedSize={savedSize}
      onSizeChange={(size) => updateAppSize("explorer", size)}
      isMaximized={maximized}
      onMaximize={() => maximizeApp("explorer")}
      onRestore={() => restoreApp("explorer")}
      isFocused={focused}
      onFocus={() => focusApp("explorer")}
    >
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-muted/30">
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={goUp}
            disabled={!canGoUp()}
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={() => refreshDirectory()}
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
          
          <div className="flex-1 flex items-center bg-muted/50 rounded-md px-3 py-1.5 overflow-x-auto">
            {showDrivesView ? (
              <span className="text-sm text-muted-foreground">
                {isWindows ? "This PC" : "Computer"}
              </span>
            ) : pathParts.length === 0 ? (
              <span className="text-sm text-muted-foreground">
                {isWindows ? "This PC" : "Computer"}
              </span>
            ) : (
              pathParts.map((part, index) => (
                <div key={index} className="flex items-center flex-shrink-0">
                  {index > 0 && <ChevronRight className="h-3 w-3 mx-1 text-muted-foreground" />}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs hover:bg-muted"
                    onClick={() => {
                      const newPath = isWindows
                        ? pathParts.slice(0, index + 1).join("/").replace(":/", ":/")
                        : "/" + pathParts.slice(0, index + 1).join("/");
                      navigateTo(newPath);
                    }}
                  >
                    {part}
                  </Button>
                </div>
              ))
            )}
          </div>

          <div className="relative w-64 flex-shrink-0">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-8 bg-muted/50 border-0"
            />
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          <div className="w-56 border-r border-border/50 bg-muted/20 overflow-y-auto flex-shrink-0">
            <div className="p-2">
              <button
                className={cn(
                  "w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm hover:bg-muted/50 transition-colors mb-1",
                  showDrivesView && "bg-muted"
                )}
                onClick={() => navigateTo("")}
              >
                <Computer className="h-4 w-4 text-blue-500 flex-shrink-0" />
                <span>{isWindows ? "This PC" : "Computer"}</span>
              </button>
              
              <div className="mt-3 px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase">
                {isWindows ? "Devices and drives" : "Storage"}
              </div>
              
              {diskInfo && (
                <div className="flex items-center justify-between px-3 py-2 border-t border-border/50 bg-muted/30 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    {diskInfo.total > 0 && (
                      <span className="text-xs text-muted-foreground">
                        {formatSize(diskInfo.used)} / {formatSize(diskInfo.total)} used
                      </span>
                    )}
                    {diskInfo.free > 0 && (
                      <span className="text-xs text-muted-foreground">
                        {formatSize(diskInfo.free)} free of {formatSize(diskInfo.total)}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 bg-muted/10">
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 bg-muted/50 hover:bg-muted"
                  onClick={() => {
                    setCreatingFolder(true);
                    setCreatingFile(false);
                    setNewItemName("");
                  }}
                >
                  <FolderPlus className="h-3 w-3 mr-1" />
                  New Folder
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 bg-muted/50 hover:bg-muted"
                  onClick={() => {
                    setCreatingFile(true);
                    setCreatingFolder(false);
                    setNewItemName("");
                  }}
                >
                  <FilePlus className="h-3 w-3 mr-1" />
                  New File
                </Button>
                {clipboard && (
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-7 bg-muted/50 hover:bg-muted"
                    onClick={handlePaste}
                  >
                    <Clipboard className="h-3 w-3 mr-1" />
                    Paste
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => setViewMode("list")}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div 
              className="flex-1 overflow-auto p-4"
              onContextMenu={handleBackgroundContextMenu}
            >
              {isLoading ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <RefreshCw className="h-6 w-6 animate-spin mr-2" />
                  Loading...
                </div>
              ) : showDrivesView ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <HardDrive className="h-12 w-12 mb-2 opacity-30" />
                  <p className="text-sm">Select a drive to browse</p>
                  {diskInfo && (
                    <div className="mt-4 text-xs">
                      <p>{formatSize(diskInfo.used)} / {formatSize(diskInfo.total)} used</p>
                      <p>{formatSize(diskInfo.free)} free</p>
                    </div>
                  )}
                </div>
              ) : filteredItems.length === 0 && !creatingFolder && !creatingFile ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <FolderOpen className="h-12 w-12 mb-2 opacity-30" />
                  <p className="text-sm">This folder is empty</p>
                </div>
              ) : viewMode === "grid" ? (
                <div className="grid grid-cols-6 gap-2">
                  {creatingFolder && (
                    <div className="flex flex-col items-center p-3 rounded-lg bg-muted">
                      <Folder className="h-10 w-10 text-amber-500 mb-1" />
                      <div className="flex items-center gap-1 w-full">
                        <Input
                          value={newItemName}
                          onChange={(e) => setNewItemName(e.target.value)}
                          placeholder="Folder name"
                          className="h-6 text-xs"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleCreateFolder();
                            if (e.key === "Escape") {
                              setCreatingFolder(false);
                              setNewItemName("");
                            }
                          }}
                        />
                        <Button
                          size="icon"
                          className="h-6 w-6"
                          onClick={handleCreateFolder}
                        >
                          <Check className="h-3 w-3" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-6 w-6"
                          onClick={() => {
                            setCreatingFolder(false);
                            setNewItemName("");
                          }}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  )}
                  {creatingFile && (
                    <div className="flex flex-col items-center p-3 rounded-lg bg-muted">
                      <File className="h-10 w-10 text-gray-500 mb-1" />
                      <div className="flex items-center gap-1 w-full">
                        <Input
                          value={newItemName}
                          onChange={(e) => setNewItemName(e.target.value)}
                          placeholder="File name"
                          className="h-6 text-xs"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleCreateFile();
                            if (e.key === "Escape") {
                              setCreatingFile(false);
                              setNewItemName("");
                            }
                          }}
                        />
                        <Button
                          size="icon"
                          className="h-6 w-6"
                          onClick={handleCreateFile}
                        >
                          <Check className="h-3 w-3" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-6 w-6"
                          onClick={() => {
                            setCreatingFile(false);
                            setNewItemName("");
                          }}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  )}
                  {filteredItems.map((item, index) => (
                    <div
                      key={index}
                      className={cn(
                        "flex flex-col items-center p-3 rounded-lg hover:bg-muted/50 transition-colors cursor-pointer",
                        selectedPath === item.path && "bg-muted",
                        clipboard?.path === item.path && clipboard.operation === "cut" && "opacity-50"
                      )}
                      onDoubleClick={() => {
                        if (item.type === "directory") {
                          navigateTo(item.path);
                        }
                      }}
                      onClick={() => setSelectedPath(item.path)}
                      onContextMenu={(e) => handleContextMenu(e, item)}
                    >
                      {renamingItem?.path === item.path ? (
                        <>
                          {item.type === "directory" ? (
                            <Folder className="h-10 w-10 text-amber-500 mb-1" />
                          ) : (
                            <div className="h-10 w-10 flex items-center justify-center">
                              {getFileIcon(item)}
                            </div>
                          )}
                          <div className="flex items-center gap-1 w-full">
                            <Input
                              value={newName}
                              onChange={(e) => setNewName(e.target.value)}
                              className="h-6 text-xs"
                              autoFocus
                              onKeyDown={(e) => {
                                if (e.key === "Enter") handleRename();
                                if (e.key === "Escape") {
                                  setRenamingItem(null);
                                  setNewName("");
                                }
                              }}
                            />
                            <Button
                              size="icon"
                              className="h-6 w-6"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRename();
                              }}
                            >
                              <Check className="h-3 w-3" />
                            </Button>
                          </div>
                        </>
                      ) : (
                        <>
                          {item.type === "directory" ? (
                            <Folder className="h-10 w-10 text-amber-500 mb-1" />
                          ) : (
                            <div className="h-10 w-10 flex items-center justify-center">
                              {getFileIcon(item)}
                            </div>
                          )}
                          <span className="text-xs text-center truncate w-full">{item.name}</span>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-0.5">
                  <div className="grid grid-cols-12 gap-2 px-3 py-2 text-xs font-medium text-muted-foreground border-b border-border/50">
                    <div className="col-span-6">Name</div>
                    <div className="col-span-2">Modified</div>
                    <div className="col-span-2">Type</div>
                    <div className="col-span-2 text-right">Size</div>
                  </div>
                  {creatingFolder && (
                    <div className="w-full grid grid-cols-12 gap-2 px-3 py-2 text-sm rounded-md bg-muted">
                      <div className="col-span-6 flex items-center gap-2">
                        <Folder className="h-4 w-4 text-amber-500" />
                        <div className="flex items-center gap-1 flex-1">
                          <Input
                            value={newItemName}
                            onChange={(e) => setNewItemName(e.target.value)}
                            placeholder="Folder name"
                            className="h-6 text-xs flex-1"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleCreateFolder();
                              if (e.key === "Escape") {
                                setCreatingFolder(false);
                                setNewItemName("");
                              }
                            }}
                          />
                          <Button
                            size="icon"
                            className="h-6 w-6"
                            onClick={handleCreateFolder}
                          >
                            <Check className="h-3 w-3" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6"
                            onClick={() => {
                              setCreatingFolder(false);
                              setNewItemName("");
                            }}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      <div className="col-span-2" />
                      <div className="col-span-2" />
                      <div className="col-span-2" />
                    </div>
                  )}
                  {creatingFile && (
                    <div className="w-full grid grid-cols-12 gap-2 px-3 py-2 text-sm rounded-md bg-muted">
                      <div className="col-span-6 flex items-center gap-2">
                        <File className="h-4 w-4 text-gray-500" />
                        <div className="flex items-center gap-1 flex-1">
                          <Input
                            value={newItemName}
                            onChange={(e) => setNewItemName(e.target.value)}
                            placeholder="File name"
                            className="h-6 text-xs flex-1"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleCreateFile();
                              if (e.key === "Escape") {
                                setCreatingFile(false);
                                setNewItemName("");
                              }
                            }}
                          />
                          <Button
                            size="icon"
                            className="h-6 w-6"
                            onClick={handleCreateFile}
                          >
                            <Check className="h-3 w-3" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6"
                            onClick={() => {
                              setCreatingFile(false);
                              setNewItemName("");
                            }}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      <div className="col-span-2" />
                      <div className="col-span-2" />
                      <div className="col-span-2" />
                    </div>
                  )}
                  {filteredItems.map((item, index) => (
                    <div
                      key={index}
                      className={cn(
                        "w-full grid grid-cols-12 gap-2 px-3 py-2 text-sm rounded-md hover:bg-muted/50 transition-colors cursor-pointer",
                        selectedPath === item.path && "bg-muted",
                        clipboard?.path === item.path && clipboard.operation === "cut" && "opacity-50"
                      )}
                      onDoubleClick={() => {
                        if (item.type === "directory") {
                          navigateTo(item.path);
                        }
                      }}
                      onClick={() => setSelectedPath(item.path)}
                      onContextMenu={(e) => handleContextMenu(e, item)}
                    >
                      {renamingItem?.path === item.path ? (
                        <>
                          <div className="col-span-6 flex items-center gap-2">
                            {getFileIcon(item)}
                            <div className="flex items-center gap-1 flex-1">
                              <Input
                                value={newName}
                                onChange={(e) => setNewName(e.target.value)}
                                className="h-6 text-xs flex-1"
                                autoFocus
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") handleRename();
                                  if (e.key === "Escape") {
                                    setRenamingItem(null);
                                    setNewName("");
                                  }
                                }}
                              />
                              <Button
                                size="icon"
                                className="h-6 w-6"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRename();
                                }}
                              >
                                <Check className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          <div className="col-span-2" />
                          <div className="col-span-2" />
                          <div className="col-span-2" />
                        </>
                      ) : (
                        <>
                          <div className="col-span-6 flex items-center gap-2 truncate">
                            {getFileIcon(item)}
                            <span className="truncate">{item.name}</span>
                          </div>
                          <div className="col-span-2 text-muted-foreground text-xs self-center">
                            {formatDate(item.modified)}
                          </div>
                          <div className="col-span-2 text-muted-foreground text-xs self-center">
                            {item.type === "directory" ? "Folder" : item.name.split(".").pop()?.toUpperCase() || "File"}
                          </div>
                          <div className="col-span-2 text-right text-muted-foreground text-xs self-center">
                            {item.type === "directory" ? "-" : formatSize(item.size)}
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between px-3 py-2 border-t border-border/50 bg-muted/30 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>{showDrivesView ? `0 drives` : `${filteredItems.length} items`}</span>
            {!showDrivesView && diskInfo && (
              <span>
                {formatSize(diskInfo.free)} free of {formatSize(diskInfo.total)}
              </span>
            )}
            {clipboard && (
              <span className="text-blue-500">
                {clipboard.operation === "copy" ? "Copied" : "Cut"}: {clipboard.path.split("/").pop()}
              </span>
            )}
          </div>
          {selectedPath && (
            <span>Selected: {selectedPath}</span>
          )}
        </div>
      </div>
    </AppWindow>
  );
}
