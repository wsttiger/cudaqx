.. _realtime_pipeline_api:

Realtime Pipeline API
=====================

The realtime pipeline API provides a framework for building low-latency QEC
decoding pipelines that combine GPU inference (e.g. TensorRT) with CPU
post-processing (e.g. PyMatching MWPM).  All types live in the
``cudaq::qec::realtime::experimental`` namespace and are declared in
``cudaq/qec/realtime/pipeline.h``.

.. note::

   This API is experimental and subject to change.


Configuration
-------------

.. class:: core_pinning

   CPU core affinity settings for pipeline threads.

   :param dispatcher: Core for the host dispatcher thread (-1 to disable pinning).
   :param consumer: Core for the consumer (completion) thread (-1 to disable pinning).
   :param worker_base: Base core for worker threads. Workers pin to
       base, base+1, etc. (-1 to disable pinning).


.. class:: pipeline_stage_config

   Configuration for a single pipeline stage.

   :param num_workers: Number of GPU worker threads (max 64). Default: 8.
   :param num_slots: Number of ring buffer slots. Default: 32.
   :param slot_size: Size of each ring buffer slot in bytes. Default: 16384.
   :param cores: CPU core affinity settings (``core_pinning``).
   :param external_ringbuffer: When non-null, the pipeline uses this
       caller-owned ring buffer (``cudaq_ringbuffer_t*``) instead of
       allocating its own.  The caller is responsible for lifetime.
       ``ring_buffer_injector`` is unavailable in this mode.


GPU Stage
---------

.. class:: gpu_worker_resources

   Per-worker GPU resources returned by the ``gpu_stage_factory``.

   Each worker owns a captured CUDA graph, a dedicated stream, and optional
   pre/post launch callbacks for DMA staging or result extraction.

   :param graph_exec: Instantiated CUDA graph (``cudaGraphExec_t``).
   :param stream: Dedicated CUDA stream (``cudaStream_t``).
   :param pre_launch_fn: Optional callback invoked before graph launch.
   :param pre_launch_data: Opaque user data for ``pre_launch_fn``.
   :param post_launch_fn: Optional callback invoked after graph launch.
   :param post_launch_data: Opaque user data for ``post_launch_fn``.
   :param function_id: RPC function ID that this worker handles.
   :param user_context: Opaque user context passed to the CPU stage callback.


.. type:: gpu_stage_factory

   ``std::function<gpu_worker_resources(int worker_id)>``

   Factory called once per worker during ``start()``.  Returns the GPU
   resources for the given worker index.


CPU Stage
---------

.. class:: cpu_stage_context

   Context passed to the CPU stage callback for each completed GPU workload.

   :param worker_id: Index of the worker thread.
   :param origin_slot: Ring buffer slot that originated this request.
   :param gpu_output: Pointer to GPU inference output (nullptr in poll mode).
   :param gpu_output_size: Size of GPU output in bytes.
   :param response_buffer: Destination buffer for the RPC response.
   :param max_response_size: Maximum bytes writable to ``response_buffer``.
   :param user_context: Opaque context from ``gpu_worker_resources``.


.. type:: cpu_stage_callback

   ``std::function<size_t(const cpu_stage_context &ctx)>``

   Returns the number of bytes written into ``response_buffer``.  Special
   return values:

   - **0**: No GPU result ready yet; the pipeline will poll again.
   - **DEFERRED_COMPLETION** (``SIZE_MAX``): Release the worker immediately
     but defer slot completion.  The caller must call
     ``realtime_pipeline::complete_deferred(slot)`` once the deferred work
     finishes.


Completion
----------

.. class:: completion

   Metadata for a completed (or errored) pipeline request.

   :param request_id: Original request ID from the RPC header.
   :param slot: Ring buffer slot that held this request.
   :param success: True if the request completed without CUDA errors.
   :param cuda_error: CUDA error code (0 on success).


.. type:: completion_callback

   ``std::function<void(const completion &c)>``

   Invoked by the consumer thread for each completed or errored request.


Ring Buffer Injector
--------------------

.. class:: ring_buffer_injector

   Writes RPC-framed requests into the pipeline's ring buffer, simulating
   FPGA DMA deposits.  Created via ``realtime_pipeline::create_injector()``.
   The parent ``realtime_pipeline`` must outlive the injector.

   Not available when the pipeline is configured with an external ring buffer
   (``pipeline_stage_config::external_ringbuffer != nullptr``).

   .. method:: bool try_submit(uint32_t function_id, const void *payload, size_t payload_size, uint64_t request_id)

      Try to submit a request without blocking.

      :param function_id: RPC function identifier.
      :param payload: Pointer to payload data.
      :param payload_size: Payload size in bytes.
      :param request_id: Caller-assigned request identifier.
      :return: True if accepted, false if all slots are busy.

   .. method:: void submit(uint32_t function_id, const void *payload, size_t payload_size, uint64_t request_id)

      Submit a request, spinning until a slot becomes available.

      :param function_id: RPC function identifier.
      :param payload: Pointer to payload data.
      :param payload_size: Payload size in bytes.
      :param request_id: Caller-assigned request identifier.

   .. method:: uint64_t backpressure_stalls() const

      :return: Cumulative number of times ``submit()`` had to spin-wait.


Pipeline
--------

.. class:: realtime_pipeline

   Orchestrates GPU inference and CPU post-processing for low-latency
   realtime QEC decoding.

   The pipeline manages a ring buffer, a host dispatcher thread, per-worker
   GPU streams with captured CUDA graphs, optional CPU worker threads, and a
   consumer thread for completion signaling.  It supports both an internal
   ring buffer (for software testing via ``ring_buffer_injector``) and an
   external ring buffer (for FPGA RDMA).

   **Lifecycle:**

   1. Construct with ``pipeline_stage_config``
   2. Register callbacks: ``set_gpu_stage()``, ``set_cpu_stage()`` (optional),
      ``set_completion_handler()`` (optional)
   3. Call ``start()`` to spawn threads
   4. Submit requests via ``ring_buffer_injector`` or external FPGA DMA
   5. Call ``stop()`` to shut down

   .. method:: realtime_pipeline(const pipeline_stage_config &config)

      Construct a pipeline and allocate ring buffer resources.

      :param config: Stage configuration.

   .. method:: void set_gpu_stage(gpu_stage_factory factory)

      Register the GPU stage factory.  Must be called before ``start()``.

      :param factory: Callback returning ``gpu_worker_resources`` per worker.

   .. method:: void set_cpu_stage(cpu_stage_callback callback)

      Register the CPU worker callback.  Must be called before ``start()``.
      If not set, the pipeline operates in GPU-only mode with completion
      signaled via ``cudaLaunchHostFunc``.

      :param callback: CPU stage processing function.

   .. method:: void set_completion_handler(completion_callback handler)

      Register the completion callback.  Must be called before ``start()``.

      :param handler: Function called for each completed request.

   .. method:: void start()

      Allocate resources, build dispatcher config, and spawn all threads.

   .. method:: void stop()

      Signal shutdown, join all threads, and free resources.

   .. method:: ring_buffer_injector create_injector()

      Create a software injector for testing without FPGA hardware.

      :return: A ``ring_buffer_injector`` bound to this pipeline.
      :raises std::logic_error: If the pipeline uses an external ring buffer.

   .. method:: Stats stats() const

      Thread-safe, lock-free statistics snapshot.

      :return: Current ``Stats`` struct.

   .. method:: void complete_deferred(int slot)

      Signal that deferred processing for a slot is complete.  Call from any
      thread after the CPU stage callback returned ``DEFERRED_COMPLETION``.

      :param slot: Ring buffer slot index to complete.

   .. method:: ring_buffer_bases ringbuffer_bases() const

      :return: Host and device base addresses of the RX data ring.

   .. class:: Stats

      Pipeline throughput and backpressure statistics.

      :param submitted: Total requests submitted to the ring buffer.
      :param completed: Total requests that completed (success or error).
      :param dispatched: Total packets dispatched by the host dispatcher.
      :param backpressure_stalls: Cumulative producer backpressure stalls.

   .. class:: ring_buffer_bases

      Host and device base addresses of the RX data ring.

      :param rx_data_host: Host-mapped base pointer.
      :param rx_data_dev: Device-mapped base pointer.
