from typing import Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class KMeansBatch(nn.Module):
    """Batch k-means over tensors shaped as (batch, num_points, dim)."""

    def __init__(self, k: int, num_iters: int) -> None:
        super().__init__()
        self.k = k
        self.num_iters = num_iters

    def _init_centers(self, x: torch.Tensor) -> torch.Tensor:
        """Sample initial centers from the input points in each batch item."""

        batch_size, num_points, _ = x.shape
        if self.k > num_points:
            raise ValueError("k must be smaller than or equal to the number of points")

        random_order = torch.rand(batch_size, num_points, device=x.device).argsort(dim=1)
        center_indices = random_order[:, : self.k]
        gather_index = center_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.gather(x, dim=1, index=gather_index)

    def _assign_clusters(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Assign each point to its nearest center."""

        distances = torch.cdist(x, centers)
        return torch.argmin(distances, dim=-1)

    def _update_centers(
        self,
        x: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute centers and reseed empty clusters from the batch."""

        batch_size, _, dim = x.shape

        expanded_assignments = assignments.unsqueeze(-1).expand(-1, -1, dim)
        center_sums = torch.zeros(
            batch_size,
            self.k,
            dim,
            device=x.device,
            dtype=x.dtype,
        )
        center_sums.scatter_add_(dim=1, index=expanded_assignments, src=x)

        counts = torch.zeros(batch_size, self.k, device=x.device, dtype=x.dtype)
        counts.scatter_add_(dim=1, index=assignments, src=torch.ones_like(assignments, dtype=x.dtype))

        safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
        updated_centers = center_sums / safe_counts

        empty_clusters = counts.eq(0).unsqueeze(-1)
        random_point_indices = torch.randint(0, x.size(1), (batch_size, self.k), device=x.device)
        gather_index = random_point_indices.unsqueeze(-1).expand(-1, -1, dim)
        replacement_centers = torch.gather(x, dim=1, index=gather_index)
        return torch.where(empty_clusters, replacement_centers, updated_centers)

    def fit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final centers and per-point cluster assignments."""

        if x.ndim != 3:
            raise ValueError("Expected x to have shape (batch, num_points, dim)")

        centers = self._init_centers(x)

        for _ in range(self.num_iters):
            assignments = self._assign_clusters(x, centers)
            centers = self._update_centers(x, assignments)

        return centers, assignments


def project_to_2d(points: torch.Tensor, basis_source: torch.Tensor | None = None) -> torch.Tensor:
    """Project points to 2D directly or through a PCA basis for plotting."""

    if points.ndim != 2:
        raise ValueError("Expected points to have shape (num_points, dim)")

    num_points, dim = points.shape
    if dim == 1:
        zeros = torch.zeros(num_points, 1, device=points.device, dtype=points.dtype)
        return torch.cat([points, zeros], dim=1)

    if dim == 2:
        return points

    projection_basis = points if basis_source is None else basis_source
    basis_mean = projection_basis.mean(dim=0, keepdim=True)
    centered_basis = projection_basis - basis_mean
    _, _, principal_components = torch.pca_lowrank(centered_basis, q=2)
    return (points - basis_mean) @ principal_components[:, :2]


def save_cluster_plot(
    x: torch.Tensor,
    assignments: torch.Tensor,
    centers: torch.Tensor,
    output_path: str,
) -> None:
    """Save a 2D visualization for the first batch item."""

    sample_points = x[0].detach().cpu()
    sample_assignments = assignments[0].detach().cpu()
    sample_centers = centers[0].detach().cpu()

    points_2d = project_to_2d(sample_points, basis_source=sample_points)
    centers_2d = project_to_2d(sample_centers, basis_source=sample_points)

    plt.figure(figsize=(8, 6))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=sample_assignments, cmap="tab10", alpha=0.75)
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c="black",
        marker="X",
        s=160,
        linewidths=1.5,
    )
    plt.title("Batch K-Means Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(7)

    batch_size, num_points, dim = 4, 600, 8
    num_clusters = 4

    cluster_offsets = torch.tensor(
        [
            [3.0, 0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 1.0],
            [-3.0, 1.5, -1.0, 2.0, -1.5, -2.0, 1.5, -0.5],
            [0.0, -3.0, 2.0, 1.0, 2.5, -1.0, 0.0, 2.0],
            [2.5, 3.0, -2.0, 0.0, -2.5, 1.0, 2.0, -1.5],
        ]
    )
    cluster_ids = torch.randint(0, num_clusters, (batch_size, num_points))
    x = cluster_offsets[cluster_ids] + 0.45 * torch.randn(batch_size, num_points, dim)

    kmeans = KMeansBatch(k=num_clusters, num_iters=15)
    centers, assignments = kmeans.fit(x)

    save_cluster_plot(
        x,
        assignments,
        centers,
        output_path="src/k-mean-batch/plot.png",
    )