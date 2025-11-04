from flash_kmeans import FlashKMeans, batch_kmeans_Euclid
import torch
import time


def test_fit_predict_api(B: int, N: int, K: int, D: int):
    data = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    kmeans = FlashKMeans(d=D, k=K, niter=20, verbose=False, seed=42, device=torch.device("cuda:0"), use_triton=False)
    kmeans.fit_predict(data)

    start_time = time.time()
    api_labels = kmeans.fit_predict(data)
    end_time = time.time()
    api_time = end_time - start_time

    batch_kmeans_Euclid(data, n_clusters=K, max_iters=20)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start_time = time.time()
    raw_labels, _, _ = batch_kmeans_Euclid(data, n_clusters=K, max_iters=20)
    end_time = time.time()
    raw_time = end_time - start_time

    print(f"FlashKMeans API time: {api_time:.4f} seconds")
    print(f"Batch KMeans raw time: {raw_time:.4f} seconds")

    try:
        torch.testing.assert_close(api_labels, raw_labels)
    except Exception as e:
        print(e)

def test_fit_predict_api_no_batch(N: int, K: int, D: int):
    data = torch.randn(N, D, device="cuda", dtype=torch.float32)
    kmeans = FlashKMeans(d=D, k=K, niter=20, verbose=False, seed=42, device=torch.device("cuda:0"), use_triton=False)
    kmeans.fit_predict(data)

    start_time = time.time()
    api_labels = kmeans.fit_predict(data)
    end_time = time.time()
    api_time = end_time - start_time

    batch_kmeans_Euclid(data.unsqueeze(0), n_clusters=K, max_iters=20)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start_time = time.time()
    raw_labels, _, _ = batch_kmeans_Euclid(data.unsqueeze(0), n_clusters=K, max_iters=20)
    end_time = time.time()
    raw_time = end_time - start_time

    print(f"FlashKMeans API time: {api_time:.4f} seconds")
    print(f"Batch KMeans raw time: {raw_time:.4f} seconds")

    try:
        torch.testing.assert_close(api_labels, raw_labels.squeeze(0))
    except Exception as e:
        print(e)

def test_train_predict_api(B: int, N: int, K: int, D: int):
    data = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    kmeans = FlashKMeans(d=D, k=K, niter=20, verbose=False, seed=42, device=torch.device("cuda:0"), use_triton=False)
    kmeans.train(data)

    start_time = time.time()
    kmeans.train(data)
    api_labels = kmeans.predict(data)
    end_time = time.time()
    api_time = end_time - start_time

    batch_kmeans_Euclid(data, n_clusters=K, max_iters=20)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start_time = time.time()
    raw_labels, _, _ = batch_kmeans_Euclid(data, n_clusters=K, max_iters=20)
    end_time = time.time()
    raw_time = end_time - start_time

    print(f"FlashKMeans API time: {api_time:.4f} seconds")
    print(f"Batch KMeans raw time: {raw_time:.4f} seconds")

    try:
        torch.testing.assert_close(api_labels, raw_labels)
    except Exception as e:
        print(e)

if __name__ == "__main__":

    test_fit_predict_api_no_batch(N=742560, K=1000, D=128)
    test_train_predict_api(B=32, N=74256, K=1000, D=128)
    test_fit_predict_api(B=32, N=74256, K=1000, D=128)