import { Compass } from "lucide-react";
import { Link } from "react-router";

import { Button, EmptyState } from "@/components/ui";

export function NotFoundPage() {
  return (
    <div className="grid min-h-dvh place-items-center px-4">
      <EmptyState
        icon={<Compass className="size-5" />}
        title="Page not found"
        action={
          <Link to="/">
            <Button variant="secondary">Back to home</Button>
          </Link>
        }
      >
        That link doesn't go anywhere.
      </EmptyState>
    </div>
  );
}
