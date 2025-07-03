# Google Gemini Message Handling Guide

## Overview

This document outlines the specific message formatting requirements and fixes implemented for Google Gemini (GenAI) integration in the AI-Powered Amazon Product Assistant. The Google GenAI client has unique message structure requirements that differ from OpenAI/Groq standards.

## 🚨 Issue Resolution

### **Problem Encountered**
```
ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': '* GenerateContentRequest.contents[1].parts: contents.parts must not be empty.\n* GenerateContentRequest.contents[3].parts: contents.parts must not be empty.\n* GenerateContentRequest.contents[5].parts: contents.parts must not be empty.\n* GenerateContentRequest.contents[7].parts: contents.parts must not be empty.\n* GenerateContentRequest.contents[8].parts: contents.parts must not be empty.\n', 'status': 'INVALID_ARGUMENT'}}
```

**Root Cause**: Google GenAI client was receiving improperly formatted messages with empty parts, causing API rejection.

### **Solution Implemented**
Enhanced message formatting with content validation, role conversion, and proper structure for Google GenAI compatibility.

## 🔧 Technical Implementation

### **Before (Problematic Code)**
```python
if provider == "Google":
    response = client.models.generate_content(
        model=model_name,
        contents=[message["content"] for message in messages],  # ❌ Incorrect format
        config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k
        }
    ).text
```

**Issues with Previous Implementation:**
- ❌ Only passing `message["content"]` without structure
- ❌ No validation for empty or whitespace content
- ❌ Incorrect role mapping for Google's expected format
- ❌ Missing required `parts` structure

### **After (Fixed Implementation)**
```python
if provider == "Google":
    # Format messages for Google GenAI - filter out empty content and convert roles
    google_messages = []
    for message in messages:
        if message.get("content") and message["content"].strip():  # ✅ Content validation
            # Convert OpenAI role format to Google format
            google_role = "user" if message["role"] == "user" else "model"  # ✅ Role conversion
            google_messages.append({
                "role": google_role,
                "parts": [{"text": message["content"]}]  # ✅ Proper structure
            })
    
    response = client.models.generate_content(
        model=model_name,
        contents=google_messages,  # ✅ Correctly formatted messages
        config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k
        }
    ).text
```

## 📋 Message Format Requirements

### **Google GenAI Expected Format**
```python
{
    "role": "user" | "model",        # Required: Conversation role
    "parts": [                       # Required: Array of content parts
        {
            "text": "message content"    # Required: Actual text content
        }
    ]
}
```

### **Role Mapping**
| OpenAI/Standard Format | Google GenAI Format | Description |
|------------------------|---------------------|-------------|
| `"user"`              | `"user"`            | User messages |
| `"assistant"`         | `"model"`           | AI responses |
| `"system"`            | `"model"`           | System prompts (converted to model) |

### **Content Validation Rules**
1. **Non-null Check**: `message.get("content")` ensures content exists
2. **Non-empty Check**: `message["content"].strip()` filters whitespace-only content
3. **Structure Validation**: Ensures proper `parts` array with `text` objects

## 🔍 Debugging and Troubleshooting

### **Common Error Patterns**

#### **Empty Parts Error**
```
GenerateContentRequest.contents[X].parts: contents.parts must not be empty
```
**Solution**: Implemented content validation to filter empty messages

#### **Invalid Role Error**
```
Invalid role specified in message
```
**Solution**: Role conversion from OpenAI format to Google format

#### **Missing Text Content**
```
Parts must contain text content
```
**Solution**: Proper `parts` structure with `text` field

### **Validation Checklist**
- ✅ All messages have non-empty content
- ✅ Roles are converted to Google format (`user`/`model`)
- ✅ Messages follow `parts` structure with `text` field
- ✅ No empty or whitespace-only messages included

## 🚀 Performance Considerations

### **Enhanced Monitoring Integration**
The fixed implementation integrates with Enhanced Tracing v2.0:

```python
# Performance tracking for Google provider
provider_emoji = {
    'Google': '🧠',  # Brain emoji for Gemini
    # ... other providers
}

# Provider-specific performance insights
elif llm_provider == "Google":
    if perf['llm_time_ms'] > 3000:
        st.warning("🧠 Google Gemini response time above average")
    else:
        st.info("🧠 Google Gemini performing within normal range")
```

### **Expected Performance Characteristics**
- **Typical Response Time**: 1000-3000ms
- **Model**: gemini-2.0-flash-exp
- **Parameters Supported**: temperature, max_output_tokens, top_p, top_k
- **Rate Limits**: Follow Google's API quotas

## 📊 Message Flow Architecture

```
Streamlit Interface
       ↓
Enhanced run_llm()
       ↓
call_llm_provider() 
       ↓
Google Message Formatter
       ↓
Content Validation
       ↓
Role Conversion (assistant → model)
       ↓
Parts Structure Creation
       ↓
Google GenAI API Call
       ↓
Response Processing
       ↓
Enhanced Tracing v2.0
```

## 🔒 Error Handling Strategy

### **Graceful Degradation**
```python
try:
    # Google GenAI call with proper formatting
    response = client.models.generate_content(...)
except ClientError as e:
    return {
        "status": "error",
        "error_type": "google_api_error",
        "error": f"Google API Error: {str(e)}",
        "metadata": request_metadata
    }
except Exception as e:
    return {
        "status": "error", 
        "error_type": type(e).__name__,
        "error": str(e),
        "metadata": request_metadata
    }
```

### **Fallback Mechanisms**
- **Empty Message Filtering**: Prevents API errors from malformed content
- **Role Validation**: Ensures compatible role mapping
- **Content Sanitization**: Strips problematic whitespace and validates text
- **Error Response Structure**: Maintains consistent error handling across all providers

## 📈 Integration with Enhanced Tracing v2.0

### **Provider-Specific Metrics**
- **🧠 Google Gemini** identified with brain emoji in performance monitoring
- **Response Time Tracking**: Specific baselines for Gemini performance
- **Error Classification**: Google-specific error types tracked
- **Parameter Support**: Full support for temperature, max_tokens, top_p, top_k

### **Business Intelligence Integration**
- User type classification works seamlessly with Gemini responses
- Intent analysis compatible with Gemini's response patterns
- Conversion tracking and satisfaction prediction integrated

## 🛠️ Development Guidelines

### **Testing Google Integration**
```python
# Test message format validation
test_messages = [
    {"role": "user", "content": "Valid message"},
    {"role": "assistant", "content": ""},  # Should be filtered out
    {"role": "user", "content": "   "},    # Should be filtered out
    {"role": "assistant", "content": "Valid response"}
]

# Expected result: Only 2 messages in google_messages
```

### **Configuration Requirements**
```bash
# Required in .env file
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Optional: Model specification
GOOGLE_MODEL=gemini-2.0-flash-exp  # Default model
```

### **Best Practices**
1. **Always validate content** before sending to Google API
2. **Use proper role mapping** for conversation context
3. **Handle empty messages gracefully** to prevent API errors
4. **Monitor performance** using provider-specific baselines
5. **Implement comprehensive error handling** for robust operation

## 📚 References

- **Google GenAI Python Client**: https://github.com/google-gemini/generative-ai-python
- **Gemini API Documentation**: https://ai.google.dev/gemini-api/docs
- **Enhanced Tracing v2.0**: `docs/WEAVE_TRACING_GUIDE.md`
- **Multi-Provider Support**: `src/chatbot-ui/streamlit_app.py`

---

**Last Updated**: January 2025  
**Version**: Enhanced Tracing v2.0 Compatible  
**Status**: ✅ Production Ready