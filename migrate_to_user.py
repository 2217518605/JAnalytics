"""一次性迁移脚本：创建用户 1234567890，并将所有 user_id=0 的数据归到该用户"""
import os
import sys
import hashlib

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.database.db import get_session, get_engine
from src.ecommerce.models import User, SalesData, ReportFile, Conversation, GeneratedImage
from sqlalchemy import text

USERNAME = "1234567890"
PASSWORD = "1234567890"


def make_pwd(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def main():
    db = get_session()
    try:
        # 1. 查找或创建用户
        user = db.query(User).filter(User.username == USERNAME).first()
        if user:
            print(f"[OK] 用户 {USERNAME} 已存在 (id={user.id})，将更新密码")
            user.password_hash = make_pwd(PASSWORD)
            db.commit()
        else:
            user = User(
                username=USERNAME,
                password_hash=make_pwd(PASSWORD),
                display_name=USERNAME,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"[OK] 已创建用户 {USERNAME} (id={user.id})")

        uid = user.id
        print(f"     user_id = {uid}")

        tables = [
            ("ec_sales_data", SalesData),
            ("ec_report_file", ReportFile),
            ("ec_conversation", Conversation),
            ("ec_generated_image", GeneratedImage),
        ]

        for table_name, model in tables:
            cnt = db.query(model).filter(model.user_id == 0).count()
            if cnt > 0:
                db.execute(
                    text(f"UPDATE {table_name} SET user_id = :uid WHERE user_id = 0"),
                    {"uid": uid},
                )
                print(f"     {model.__tablename__}: {cnt} 条 -> user_id={uid}")

        db.commit()
        print(f"\n[DONE] 迁移完成！所有历史数据已归属到用户 {USERNAME}")
        print(f"       登录用户名: {USERNAME}")
        print(f"       登录密码:   {PASSWORD}")

    except Exception as e:
        db.rollback()
        print(f"[FAIL] {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
