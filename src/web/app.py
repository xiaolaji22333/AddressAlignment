from configuration import config
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from web.service import address_alignment


# 定义一个应用
app = FastAPI()
# 将静态文件挂载到应用当中
app.mount("/static", StaticFiles(directory="templates"), name="static")

# 定义schema信息，也就是和前端页面交互时的标准化接口的结构，分为两部分，一部分是输入，也就是xxxxRequest，
# 另外一部分是输出，也就是后端程序员给到前端的接口标准结构
class AddressAlignmentRequest(BaseModel):
    message: str = Field(description="地址文本信息")


class AddressAlignmentResponse(BaseModel):
    province: str | None = Field(description="省份")
    city: str | None = Field(description="城市")
    district: str | None = Field(description="区县")
    devzone: str | None = Field(description="区县")
    town: str | None = Field(description="乡/镇/街道")
    detail: str | None = Field(description="详细地址")


@app.get("/")
async def homepage():
    # 使用首页模板
    return FileResponse("templates/index.html")


@app.post("/address_alignment") # 这里是路径函数，定义了接口路径所对应的业务逻辑
async def handle_message(message: AddressAlignmentRequest):
    user_message = message.message
    # 调用个人定义的模块，获取到数据结果
    #adress是一个字典，里面有省份，城市，区县，街道，详细地址等信息
    address = address_alignment(user_message)
    # 封装成schema
    return AddressAlignmentResponse(
        province=address["省份"],
        city=address["城市"],
        district=address["区县"],
        devzone=address["区县"],
        town=address["街道"],
        detail=address["详细地址"]
    )



if __name__ == '__main__':
    # uvicorn是一个基于python的高性能服务器，是运行fastapi的引擎
    uvicorn.run("src.web.app:app",
                host="127.0.0.1",
                port=8888,
                reload=True,
                reload_dirs=['./templates'],
                workers=1)
