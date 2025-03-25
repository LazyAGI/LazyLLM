import lazyllm
from lazyllm.common import compile_func, package
from lazyllm.tools.http_request import HttpRequest
from lazyllm.configs import LIGHTENGINE_DB_KEY
from lazyllm.tools import SqlManager
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta
from enum import Enum
import requests

class AuthenticationFailedError(Exception):
    def __init__(self, message="Authentication failed for the given user and tool."):
        self.message = message
        super().__init__(self.message)

class TokenExpiredError(Exception):
    """Access token expired"""
    pass

class TokenRefreshError(Exception):
    """Access key request failed"""
    pass

class AuthType(Enum):
    SERVICE_API = "service_api"
    OAUTH = "oauth"
    OIDC = "oidc"

class HttpTool(HttpRequest):
    def __init__(self,
                 method: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict[str, str]] = None,
                 code_str: Optional[str] = None,
                 vars_for_code: Optional[Dict[str, Any]] = None,
                 outputs: Optional[List[str]] = None,
                 extract_from_result: Optional[bool] = None,
                 authentication_type: Optional[str] = None,
                 tool_id: Optional[str] = None,
                 user_id: Optional[str] = None):
        super().__init__(method, url, '', headers, params, body, timeout, proxies)
        self._has_http = True if url else False
        self._compiled_func = (compile_func(code_str, vars_for_code) if code_str else
                               (lambda x: json.loads(x['content'])) if self._has_http else None)
        self._outputs, self._extract_from_result = outputs, extract_from_result
        if extract_from_result:
            assert outputs, 'Output information is necessary to extract output parameters'
            assert len(outputs) == 1, 'When the number of outputs is greater than 1, no manual setting is required'
        self.token_type = authentication_type
        self._tool_id = tool_id
        self._user_id = user_id
        self._key_db_connect_message = lazyllm.globals.get(LIGHTENGINE_DB_KEY)
        self._sql_manager = SqlManager(
            db_type=self._key_db_connect_message['db_type'],
            user=self._key_db_connect_message.get('user', None),
            password=self._key_db_connect_message.get('password', None),
            host=self._key_db_connect_message.get('host', None),
            port=self._key_db_connect_message.get('port', None),
            db_name=self._key_db_connect_message['db_name'],
            options_str=self._key_db_connect_message.get('options_str', None),
            tables_info_dict=self._key_db_connect_message.get('tables_info_dict', None),
        )
        self._default_expired_days = 3

    def _process_api_key(self, headers, params):
        if not self.token_type:
            return headers, params
        if self.token_type == AuthType.SERVICE_API.value:
            if self._location == "header":
                headers[self._param_name] = self._token if self._token.startswith("Bearer") \
                    else "Bearer " + self._token
            elif self._location == "query":
                params[self._param_name] = self._token
            else:
                raise TypeError("The Service API authentication type only supports ['header', 'query'], "
                                f"not {self._location}.")
        elif self.token_type == AuthType.OAUTH.value:
            headers['Authorization'] = f"Bearer {self._token}"
        else:
            raise TypeError("Currently, tool authentication only supports ['service_api', 'oauth'] types, "
                            f"and does not support {self.token_type} type.")
        return headers, params

    def valid_key(self):
        table_name = self._key_db_connect_message.get('tables_info_dict', {}).get('tables', [])[0]['name']
        SQL_SELECT = (
            f"SELECT id, tool_id, endpoint_url, client_id, client_secret, user_id, location, param_name, token, "
            f"refresh_token, token_type, expires_at FROM {table_name} "
            f"WHERE tool_id = '{self._tool_id}' AND is_auth_success = 1 AND token_type = '{self.token_type}'"
        )
        ret = self._fetch_valid_key(SQL_SELECT + " AND is_share = 1")
        if not ret:
            ret = self._fetch_valid_key(SQL_SELECT + f" AND user_id = '{self._user_id}'")
            if not ret:
                raise AuthenticationFailedError(f"Authentication failed for user_id='{self._user_id}' and "
                                                f"tool_id='{self._tool_id}'")

        if self.token_type == AuthType.SERVICE_API.value:
            # self._process_authentication_key(ret['token'], ret['location'], ret['param_name'])
            self._token = ret['token']
            self._location = ret['location']
            self._param_name = ret['param_name']
        elif self.token_type == AuthType.OAUTH.value:
            # self._process_authentication_key(self._validate_and_refresh_token(
            #     id=ret['id'],
            #     client_id=ret['client_id'],
            #     client_secret=ret['client_secret'],
            #     endpoint_url=ret['endpoint_url'],
            #     token=ret['token'],
            #     refresh_token=ret['refresh_token'],
            #     expires_at=datetime.strptime(ret['expires_at'], "%Y-%m-%d %H:%M:%S")))
            self._token = self._validate_and_refresh_token(
                id=ret['id'],
                client_id=ret['client_id'],
                client_secret=ret['client_secret'],
                endpoint_url=ret['endpoint_url'],
                token=ret['token'],
                refresh_token=ret['refresh_token'],
                expires_at=datetime.strptime(ret['expires_at'], "%Y-%m-%d %H:%M:%S"),
                table_name=table_name)
        elif self.token_type == AuthType.OIDC.value:
            raise TypeError("OIDC authentication is not currently supported.")
        else:
            raise TypeError("The authentication type only supports ['no authentication', 'service_api', "
                            f"'oauth', 'oidc'], and does not support type {self.token_type}.")

    def _fetch_valid_key(self, query: str):
        ret = self._sql_manager.execute_query(query)
        ret = json.loads(ret)
        return ret[0] if ret else None

    def _validate_and_refresh_token(self, id: int, client_id: str, client_secret: str, endpoint_url: str,
                                    token: str, refresh_token: str, expires_at: datetime, table_name):
        now = datetime.now()
        # 1、Access token has not expired
        if now < expires_at:
            if not refresh_token:
                # Update only the expiration time
                new_expires_at = now + timedelta(days=self._default_expired_days)
                self._sql_manager.execute_commit(f"UPDATE {table_name} SET expires_at = "
                                                 f"'{new_expires_at.strftime('%Y-%m-%d %H:%M:%S')}' WHERE id = {id}")
            return token

        # 2、Access token expired
        if not refresh_token:
            raise TokenExpiredError("Access key has expired, and no refresh key was provided.")

        # 3、Request a new access token with the refresh_token
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {client_secret}"}
        data = {"client_id": '{client_id}', "grant_type": "refresh_token", "refresh_token": '{refresh_token}'}
        with requests.post(endpoint_url, json=data, headers=headers) as r:
            if r.status_code != 200:
                raise TokenRefreshError(f"Request failed, status code: {r.status_code}, message: {r.text}")

            data = r.json()
            new_token = data.get("access_token")
            new_refresh_token = data.get("refresh_token")
            new_expires_at = data.get("expires_in")

            # update db
            self._sql_manager.execute_commit(f"UPDATE {table_name} SET token = '{new_token}', refresh_token = "
                                             f"'{new_refresh_token}', expires_at = '{new_expires_at}' where id = {id}")
            return new_token

    def _get_result(self, res):
        if self._extract_from_result or (isinstance(res, dict) and len(self._outputs) > 1):
            assert isinstance(res, dict), 'The result of the tool should be a dict type'
            r = package(res.get(key) for key in self._outputs)
            return r[0] if len(r) == 1 else r
        if len(self._outputs) > 1:
            assert isinstance(res, (tuple, list)), 'The result of the tool should be tuple or list'
            assert len(res) == len(self._outputs), 'The number of outputs is inconsistent with expectations'
            return package(res)
        return res

    def forward(self, *args, **kwargs):
        if not self._compiled_func: return None
        if self._has_http:
            res = super().forward(*args, **kwargs)
            if int(res['status_code']) >= 400:
                raise RuntimeError(f'HttpRequest error, status code is {res["status_code"]}.')
            args, kwargs = (res,), {}
        res = self._compiled_func(*args, **kwargs)
        return self._get_result(res) if self._outputs else res
