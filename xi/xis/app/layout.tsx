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

import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";
import { SidebarProvider } from "@/components/layout/sidebar-context";
import { AppsProvider } from "@/components/layout/apps-context";
import { ContextMenuProvider } from "@/components/layout/context-menu";
import { MainLayout } from "@/components/layout/main-layout";
import { AppBootstrap } from "@/components/app-bootstrap";

export const metadata: Metadata = {
  title: "Xi Studio",
  description: "LLM Workstation for training, inference, and data management",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <Providers>
          <AppBootstrap>
            <SidebarProvider>
              <AppsProvider>
                <ContextMenuProvider>
                  <MainLayout>{children}</MainLayout>
                </ContextMenuProvider>
              </AppsProvider>
            </SidebarProvider>
          </AppBootstrap>
        </Providers>
      </body>
    </html>
  );
}
