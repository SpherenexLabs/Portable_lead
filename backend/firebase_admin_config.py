"""
firebase_admin_config.py — NOT USED
=====================================
Firebase writes are handled via the Realtime Database REST API directly in app.py.
No service account or firebase-admin SDK is required.

The database URL from your Firebase project config is enough:
  https://diet-planner-3bdf3-default-rtdb.firebaseio.com

Make sure your Firebase Realtime Database rules allow writes at /Portable_lead/ML_Result.
For development, set rules to:
  {
    "rules": {
      ".read": true,
      ".write": true
    }
  }
"""
