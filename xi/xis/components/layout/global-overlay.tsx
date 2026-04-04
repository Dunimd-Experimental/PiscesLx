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

import { X } from "lucide-react";
import "@/styles/components/global-overlay.css";

interface GlobalOverlayProps {
  layout?: "split" | "full";
  leftContent?: React.ReactNode;
  rightContent?: React.ReactNode;
  children?: React.ReactNode;
  showClose?: boolean;
  onClose?: () => void;
}

function GlobalOverlay({
  layout = "full",
  leftContent,
  rightContent,
  children,
  showClose = false,
  onClose,
}: GlobalOverlayProps) {
  const handleClose = () => {
    if (onClose) {
      onClose();
    }
  };

  return (
    <div className="global-overlay">
      <div className={`global-overlay__panel global-overlay__panel--${layout}`}>
        {showClose && (
          <button
            className="global-overlay__close"
            onClick={handleClose}
            aria-label="Close"
          >
            <X />
          </button>
        )}

        {layout === "split" && (
          <>
            {leftContent && (
              <div className="global-overlay__left">{leftContent}</div>
            )}
            {rightContent && (
              <div className="global-overlay__right">{rightContent}</div>
            )}
          </>
        )}

        {layout === "full" && children}
      </div>
    </div>
  );
}

export { GlobalOverlay };
export type { GlobalOverlayProps };
