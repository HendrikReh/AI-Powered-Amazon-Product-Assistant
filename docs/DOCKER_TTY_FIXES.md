# Docker TTY Issues Fixed

This document explains the Docker TTY (terminal) compatibility fixes implemented in the AI-Powered Amazon Product Assistant project to ensure smooth containerized deployment.

## Overview

TTY issues in Docker commonly manifest as:
- Applications hanging waiting for user input
- Permission denied errors
- Services failing to start properly
- Interactive prompts causing container failures

The implemented fixes ensure the application runs smoothly in containerized environments without requiring an interactive terminal, making it suitable for production deployment scenarios like cloud platforms, CI/CD pipelines, and orchestration systems.

## 1. Non-Root User Configuration

**Problem**: Running containers as root creates security vulnerabilities and permission issues.

**Solution**: Create a dedicated system user with proper permissions.

```dockerfile
# Create non-root user with proper home directory and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app/.streamlit && \
    chown -R app:app /app && \
    chown -R app:app /home/app

# Switch to non-root user
USER app

# Set HOME environment variable
ENV HOME=/home/app
```

**Benefits**:
- Prevents permission issues
- Reduces security attack surface
- Follows Docker security best practices
- Ensures proper file ownership

## 2. Streamlit Headless Configuration

**Problem**: Streamlit tries to open a browser and gather usage stats by default, which fails in containerized environments.

**Solution**: Configure Streamlit for headless operation.

```dockerfile
# Configure Streamlit to disable usage stats and set proper config
RUN echo '[browser]\ngatherUsageStats = false\n[server]\nheadless = true\n' > /home/app/.streamlit/config.toml
```

**Configuration Details**:
- `headless = true`: Prevents browser auto-launch attempts
- `gatherUsageStats = false`: Disables telemetry that may require TTY interaction

**Benefits**:
- Prevents TTY-related errors when no terminal is attached
- Ensures the app runs properly in container environments
- Avoids hanging on browser launch attempts

## 3. Weave Tracing Compatibility

**Problem**: W&B (Weights & Biases) authentication can prompt for interactive input, causing containers to hang.

**Solution**: Use non-interactive authentication with proper error handling.

```python
if config.WANDB_API_KEY:
    try:
        import wandb
        # Login to W&B with API key before initializing Weave
        wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
        weave.init(project_name="Bootcamp")
        st.sidebar.success("🔍 Weave tracing enabled")
    except Exception as e:
        st.sidebar.error(f"❌ Weave initialization failed: {str(e)}")
        st.sidebar.info("Continuing without tracing...")
```

**Key Parameters**:
- `force=True`: Prevents interactive prompts
- `anonymous="never"`: Ensures API key authentication
- Error handling: Graceful degradation when tracing unavailable

**Benefits**:
- Prevents interactive prompts that would fail in TTY-less environments
- Allows application to continue functioning even if tracing fails
- Provides clear feedback about tracing status

## 4. Server Address Binding

**Problem**: Default Streamlit configuration may only bind to localhost, making the service inaccessible from outside the container.

**Solution**: Bind to all network interfaces.

```dockerfile
CMD ["streamlit", "run", "src/chatbot-ui/streamlit_app.py", "--server.address=0.0.0.0"]
```

**Benefits**:
- Makes the service accessible from outside the container
- Essential for containerized deployments
- Enables proper port forwarding and load balancing

## 5. Environment Optimization

**Additional Configuration**: Performance and compatibility optimizations.

```dockerfile
# Enable bytecode compilation and Python optimization
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONOPTIMIZE=1
ENV UV_LINK_MODE=copy

# Set Python path to include the src directory for imports
ENV PYTHONPATH="/app/src:$PYTHONPATH"
```

**Benefits**:
- Faster startup times through bytecode compilation
- Optimized Python execution
- Proper module resolution

## Docker Commands

### Build the Container
```bash
make build-docker-streamlit
# or
docker build -t streamlit-app:latest .
```

### Run the Container
```bash
make run-docker-streamlit
# or
docker run -v "${PWD}/.env:/app/.env" -p 8501:8501 streamlit-app:latest
```

## Verification

To verify the fixes work correctly:

1. **Build and run the container**:
   ```bash
   docker build -t streamlit-app:latest .
   docker run -p 8501:8501 streamlit-app:latest
   ```

2. **Check for TTY-related errors** in the container logs:
   ```bash
   docker logs <container_id>
   ```

3. **Verify service accessibility**:
   - Open browser to `http://localhost:8501`
   - Confirm Streamlit interface loads properly
   - Test LLM functionality and configuration options

## Common TTY Issues Prevented

| Issue | Symptom | Fix Applied |
|-------|---------|-------------|
| Interactive prompts | Container hangs indefinitely | `force=True` in wandb.login() |
| Browser launch attempts | "No display" errors | `headless = true` in Streamlit config |
| Permission errors | File access denied | Non-root user with proper ownership |
| Service binding | Cannot access from host | `--server.address=0.0.0.0` |
| Usage stats collection | Network/TTY timeouts | `gatherUsageStats = false` |

## Production Deployment Considerations

These fixes enable deployment in various containerized environments:

- **Cloud Platforms**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Kubernetes**: Pods without TTY allocation
- **CI/CD Pipelines**: Automated testing and deployment
- **Docker Compose**: Multi-service orchestration
- **Serverless**: Container-based serverless functions

The application will run reliably in any of these environments without requiring interactive terminal access. 