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

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import {
    LayoutDashboard,
    Brain,
    MessageSquare,
    Database,
    Cpu,
    Play,
    ChevronLeft,
    ChevronRight,
    Settings,
} from "lucide-react";

const navigation = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { name: "Training", href: "/training", icon: Brain },
    { name: "Inference", href: "/inference", icon: MessageSquare },
    { name: "Data", href: "/data", icon: Database },
    { name: "Models", href: "/models", icon: Cpu },
    { name: "Runs", href: "/runs", icon: Play },
];

const bottomNavigation = [
    { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    return (
        <aside
            className={cn(
                "flex flex-col border-r bg-card transition-all duration-300",
                collapsed ? "w-16" : "w-64"
            )}
        >
            <div className="flex h-16 items-center justify-between border-b px-4">
                {!collapsed && (
                    <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg gradient-primary">
                            <span className="text-lg font-bold text-white">P</span>
                        </div>
                        <span className="text-lg font-semibold text-gradient">PLx Studio</span>
                    </div>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setCollapsed(!collapsed)}
                    className="h-8 w-8"
                >
                    {collapsed ? (
                        <ChevronRight className="h-4 w-4" />
                    ) : (
                        <ChevronLeft className="h-4 w-4" />
                    )}
                </Button>
            </div>

            <nav className="flex-1 space-y-1 p-2">
                {navigation.map((item) => {
                    const isActive = pathname.startsWith(item.href);
                    const Icon = item.icon;

                    return (
                        <Tooltip key={item.name} delayDuration={0}>
                            <TooltipTrigger asChild>
                                <Link
                                    href={item.href}
                                    className={cn(
                                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                                        isActive
                                            ? "bg-primary/10 text-primary"
                                            : "text-muted-foreground hover:bg-muted hover:text-foreground",
                                        collapsed && "justify-center px-2"
                                    )}
                                >
                                    <Icon className="h-5 w-5 flex-shrink-0" />
                                    {!collapsed && <span>{item.name}</span>}
                                </Link>
                            </TooltipTrigger>
                            {collapsed && (
                                <TooltipContent side="right" className="font-medium">
                                    {item.name}
                                </TooltipContent>
                            )}
                        </Tooltip>
                    );
                })}
            </nav>

            <div className="mt-auto border-t p-2">
                {bottomNavigation.map((item) => {
                    const isActive = pathname.startsWith(item.href);
                    const Icon = item.icon;

                    return (
                        <Tooltip key={item.name} delayDuration={0}>
                            <TooltipTrigger asChild>
                                <Link
                                    href={item.href}
                                    className={cn(
                                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                                        isActive
                                            ? "bg-primary/10 text-primary"
                                            : "text-muted-foreground hover:bg-muted hover:text-foreground",
                                        collapsed && "justify-center px-2"
                                    )}
                                >
                                    <Icon className="h-5 w-5 flex-shrink-0" />
                                    {!collapsed && <span>{item.name}</span>}
                                </Link>
                            </TooltipTrigger>
                            {collapsed && (
                                <TooltipContent side="right" className="font-medium">
                                    {item.name}
                                </TooltipContent>
                            )}
                        </Tooltip>
                    );
                })}
            </div>
        </aside>
    );
}
