#pragma once

namespace nacho {

struct Level;
struct Format;
struct TensorType;

struct Expr;
struct Add;
struct Bc;
struct Mul;
struct Sum;
struct Tensor;

struct Seq;
struct Index;
struct Intersect;
struct Union;
struct Universe;

struct cExpr;
struct cAdd;
struct cMul;
struct cTensor;

struct CIN;
struct Accumulate;
struct Assign;
struct Forall;
struct Sequence;
struct Where;

namespace llir {

struct lType;
struct Generic_t;
struct Int_t;
struct Float_t;
struct Ptr_t;

struct lExpr;
struct lBinOp;
struct lConst;
struct lLoad;
struct lVar;

struct lStmt;
struct Declare;
struct IfElse;
struct Return;
struct Sequence;
struct Store;
struct While;

struct Function;

} // namespace llir

} // namespace nacho
