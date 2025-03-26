# AIcons Framework: Tools and Limbs

## Core Concepts

### Tools

A tool in the AIcons framework is an external piece of software that helps the AIcon to change the environment somewhere else. Tools are specific implementations that interact with external services or systems. They are the concrete actions that an AIcon can take to affect change in the world.

Examples of tools:

- Meta Ads Creation Tool (creates ads using Facebook Marketing API)
- Twitter Post Tool (posts tweets using Twitter API)
- Email Sending Tool (sends emails using SMTP)

### Limbs

A limb is a piece of software that enables the usage of different tools. Limbs are like abstract interfaces or connectors that provide the capability to use various tools. They represent the general ability to interact with a type of service.

Examples of limbs:

- API Client Limb (enables making API calls to various services)
- Database Connection Limb (enables interaction with different databases)
- File System Limb (enables file operations)

## Architecture

The relationship between limbs and tools can be understood as follows:

```
Limb (Abstract Capability)
  └── Tool 1 (Specific Implementation)
  └── Tool 2 (Specific Implementation)
  └── Tool 3 (Specific Implementation)
```

For example:

```
API Client Limb
  └── Meta Ads Creation Tool
  └── Twitter Post Tool
  └── Weather API Tool
```
